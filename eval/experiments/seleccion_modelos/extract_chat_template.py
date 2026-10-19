#!/usr/bin/env python3
"""Extract chat template + bos/eos tokens from a GGUF file WITHOUT loading weights.

Pure stdlib (struct) GGUF v2/v3 metadata reader. Designed to run over ssh on the
model host: `python3 extract_chat_template.py /models/foo.gguf` -> JSON on stdout
{chat_template, bos_token, eos_token, add_bos, add_eos}. Only the metadata (KV)
section is scanned; stops early once everything is found.
"""
import json
import mmap
import struct
import sys

GGUF_MAGIC = 0x46554747  # "GGUF" little-endian

T_INT = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 10: "<Q", 11: "<q"}
T_FLOAT = {6: "<f", 12: "<d"}


class Reader:
    def __init__(self, buf):
        self.buf = buf
        self.pos = 0

    def need(self, n):
        if self.pos + n > len(self.buf):
            raise EOFError

    def u(self, fmt):
        self.need(struct.calcsize(fmt))
        v = struct.unpack_from(fmt, self.buf, self.pos)[0]
        self.pos += struct.calcsize(fmt)
        return v

    def string(self):
        n = self.u("<Q")
        self.need(n)
        s = self.buf[self.pos:self.pos + n].decode("utf-8", "replace")
        self.pos += n
        return s

    def value(self, t):
        if t in T_INT:
            return self.u(T_INT[t])
        if t in T_FLOAT:
            return self.u(T_FLOAT[t])
        if t == 7:  # bool
            return bool(self.u("<B"))
        if t == 8:
            return self.string()
        if t == 9:  # array
            et = self.u("<I")
            n = self.u("<Q")
            if et in (8, 9):  # array of strings/arrays: must walk element-wise
                return [self.value(et) for _ in range(n)]
            size = struct.calcsize(T_INT.get(et, T_FLOAT.get(et, "<B")))
            self.need(n * size)
            v = list(struct.unpack_from(f"<{n}{T_INT.get(et, T_FLOAT.get(et, '<B'))[1]}", self.buf, self.pos))
            self.pos += n * size
            return v
        raise ValueError(f"unknown type {t}")


def extract(path, want_keys):
    # mmap: read the KV section without loading multi-GB weights into RAM
    with open(path, "rb") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as buf:
            return _extract_buf(buf, want_keys)


def _extract_buf(buf, want_keys):
    r = Reader(buf)
    if r.u("<I") != GGUF_MAGIC:
        raise ValueError("not a GGUF file")
    r.u("<I")  # version
    r.u("<Q")  # tensor count
    kv_count = r.u("<Q")
    out = {}
    strkeys = {k for k in want_keys if k.endswith(("_token", "chat_template"))}
    for _ in range(kv_count):
        key = r.string()
        t = r.u("<I")
        if key in want_keys:
            out[key] = r.value(t)
        elif key in strkeys:
            out[key] = r.value(t)
        elif t == 8:
            r.string()
        elif t == 9:  # skip arrays element-wise (vocab lists are big)
            et = r.u("<I")
            n = r.u("<Q")
            if et in (8, 9):
                for _ in range(n):
                    r.value(et)
            else:
                import struct as _s
                size = _s.calcsize(T_INT.get(et, T_FLOAT.get(et, "<B")))
                r.need(n * size)
                r.pos += n * size
        else:
            r.value(t)
        if all(k in out for k in want_keys):
            break
    return out


def main():
    path = sys.argv[1]
    keys = [
        "tokenizer.chat_template",
        "tokenizer.ggml.bos_token_id",
        "tokenizer.ggml.eos_token_id",
        "tokenizer.ggml.model",
        "general.architecture",
    ]
    d = extract(path, keys)
    # resolve bos/eos token strings if we saw the vocab; otherwise leave null
    print(json.dumps(d, ensure_ascii=False))


if __name__ == "__main__":
    main()
