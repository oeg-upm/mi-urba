#!/usr/bin/env python3
"""Anota casos de un corpus con un panel de LLM vía LiteLLM (una petición por caso).

Un caso se considera ACEPTADO cuando TODAS las peticiones de todos los modelos
del panel dicen 'apto'. Los casos que no lo consiguen se registran con sus votos.

Muestreo:
  --generators A B      limita el corpus a esos modelos generadores (obligatorio
                        en la práctica: en la carpeta del dataset conviven otros
                        experimentos con ids propios que colisionan).
  --per-generator N     reúne N casos aceptados POR CADA generador.
  --target N            reúne N casos aceptados en total.

Los casos se identifican por (modelo generador, id), nunca solo por id: cada
experimento reutiliza Q0001.. y un mapa por id mezclaría modelos.

Los modelos que razonan gastan presupuesto antes de escribir: con max_tokens bajo
devuelven contenido vacío con finish_reason='length'. Por eso el presupuesto es
amplio y hay reintento con el doble de margen.

La clave se lee en tiempo de ejecución de ~/.pi/agent/local-llm.key y nunca se
escribe en el repo ni en las salidas.
"""
import argparse
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.prompt_loader import render  # noqa: E402

KEY_PATH = Path.home() / ".pi" / "agent/local-llm.key"
BASE_URL = os.environ.get("MIURBA_LLM_URL", "")


def api_key() -> str:
    return KEY_PATH.read_text(encoding="utf-8").strip()


def render_scenario(extractions: list[dict]) -> str:
    """Renderiza el ground truth del corpus (formato extraction_schema) como escenario."""
    lines = []
    for i, e in enumerate(extractions, 1):
        cat = e.get("category") or {}
        loc = e.get("location") or {}
        elems = ", ".join(f"{x.get('principal')}/{x.get('subcategory')}"
                          for x in (e.get("affected_elements") or []))
        lines.append(
            f"{i}. Problema: {cat.get('subcategory')} (categoria: {cat.get('principal')}; "
            f"zona: {loc.get('zone')}; detalle: {loc.get('detail') or '-'}); "
            f"gravedad: {e.get('urgency')}; elementos: {elems or '-'}; "
            f"riesgos: {', '.join(e.get('risks') or []) or 'ninguno'}"
        )
    return "\n".join(lines)


def call(model: str, messages: list[dict], temperature: float, max_tokens: int,
         timeout: int = 300) -> dict:
    """Una petición al endpoint. Devuelve veredicto + metadatos para el log."""
    payload = {"model": model, "messages": messages, "temperature": temperature,
               "max_tokens": max_tokens}
    t0 = time.time()
    meta = {"model": model, "max_tokens": max_tokens, "latency_s": None,
            "finish_reason": None, "completion_tokens": None,
            "reasoning_tokens": None, "content_len": None}
    try:
        req = Request(BASE_URL, json.dumps(payload).encode(),
                      {"Content-Type": "application/json",
                       "Authorization": f"Bearer {api_key()}"})
        data = json.load(urlopen(req, timeout=timeout))
        choice = data["choices"][0]
        content = (choice["message"].get("content") or "").strip()
        usage = data.get("usage") or {}
        meta.update({"finish_reason": choice.get("finish_reason"),
                     "completion_tokens": usage.get("completion_tokens"),
                     "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                     "content_len": len(content)})
        start, end = content.find("{"), content.rfind("}")
        if start < 0 or end < 0:
            raise ValueError(f"sin JSON (finish={meta['finish_reason']}, "
                             f"content_len={len(content)})")
        verdict = json.loads(content[start:end + 1])
        return {**meta, "apto": bool(verdict.get("apto")),
                "campos_graves": verdict.get("campos_graves") or [],
                "motivo": str(verdict.get("motivo") or "")[:160],
                "error": None, "latency_s": round(time.time() - t0, 1)}
    except Exception as e:  # noqa: BLE001
        return {**meta, "apto": None, "campos_graves": [], "motivo": "",
                "error": f"{type(e).__name__}: {str(e)[:140]}",
                "latency_s": round(time.time() - t0, 1)}


def ask_with_retries(model: str, messages: list[dict], temperature: float,
                     max_tokens: int, attempts: int, log) -> dict:
    """Reintenta cuando la respuesta viene vacía, truncada o con error de parseo."""
    last = None
    for attempt in range(1, attempts + 1):
        result = call(model, messages, temperature, max_tokens)
        result["attempt"] = attempt
        log(f"request\t{model}\tattempt={attempt}\tfinish={result['finish_reason']}"
            f"\ttokens={result['completion_tokens']}\treasoning={result['reasoning_tokens']}"
            f"\tcontent_len={result['content_len']}\tapto={result['apto']}"
            f"\terror={result['error']}")
        if result["error"] is None:
            return result
        last = result
        if result["finish_reason"] == "length":
            max_tokens = min(max_tokens * 2, 20000)
        time.sleep(1.5 * attempt)
    return last


def load_corpus(dataset: Path, generators: list[str]):
    """Devuelve (ids por generador, textos[(modelo,id)], gt[(modelo,id)])."""
    per_model: dict[str, list[str]] = {}
    texts, gts = {}, {}
    for model_dir in sorted(dataset.iterdir()):
        if model_dir.name not in generators:
            continue
        raw = model_dir / "raw"
        if not (raw / "complaints.jsonl").exists():
            continue
        ids = []
        for line in open(raw / "complaints.jsonl", encoding="utf-8"):
            rec = json.loads(line)
            ids.append(rec["id"])
            texts[(model_dir.name, rec["id"])] = rec["user_text"]
        for line in open(raw / "ground_truth.jsonl", encoding="utf-8"):
            rec = json.loads(line)
            gts[(model_dir.name, rec["complaint_id"])] = rec
        per_model[model_dir.name] = ids
    missing = [g for g in generators if g not in per_model]
    if missing:
        raise SystemExit(f"generadores sin datos en {dataset}: {missing}")
    return per_model, texts, gts


def main():
    global BASE_URL
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--llm-url", default=BASE_URL,
                    help="Chat-completions URL (or set MIURBA_LLM_URL)")
    ap.add_argument("--generators", nargs="*", default=[])
    ap.add_argument("--target", type=int, default=0)
    ap.add_argument("--per-generator", type=int, default=0)
    ap.add_argument("--per-model", type=int, default=3, help="peticiones por modelo juez y caso")
    ap.add_argument("--models", nargs="*",
                    default=["deepseek-v4.1-flash-1017", "zai-glm-5.3-flash"])
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--max-tokens", type=int, default=6000)
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    if not args.target and not args.per_generator:
        ap.error("indica --target o --per-generator")
    if not args.generators:
        ap.error("indica --generators (en la carpeta del dataset conviven otros experimentos)")
    if not args.llm_url:
        ap.error("indica --llm-url o define MIURBA_LLM_URL")
    BASE_URL = args.llm_url

    dataset = Path(args.dataset)
    if not dataset.is_absolute():
        dataset = (Path.cwd() / dataset).resolve()
    per_model, texts, gts = load_corpus(dataset, args.generators)
    rnd = random.Random(args.seed)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log_lock = threading.Lock()
    log_file = open(out / "run.log", "w", encoding="utf-8")

    def log(line: str) -> None:
        with log_lock:
            log_file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t{line}\n")
            log_file.flush()

    log(f"start\tdataset={dataset}\tgeneradores={','.join(args.generators)}"
        f"\ttarget={args.target}\tper_generator={args.per_generator}"
        f"\tmodelos_juez={','.join(args.models)}\tper_model={args.per_model}"
        f"\tmax_tokens={args.max_tokens}\tattempts={args.attempts}")
    for m, ids in per_model.items():
        log(f"corpus\t{m}\t{len(ids)} casos")

    f_requests = open(out / "requests.jsonl", "w", encoding="utf-8")
    f_ok = open(out / "accepted.jsonl", "w", encoding="utf-8")
    f_ko = open(out / "rejected.jsonl", "w", encoding="utf-8")

    if args.per_generator:
        quota = {m: args.per_generator for m in per_model}
        queues = {m: rnd.sample(ids, len(ids)) for m, ids in per_model.items()}
        total = args.per_generator * len(per_model)
    else:
        quota = None
        queues = {"*": rnd.sample([(m, i) for m, ids in per_model.items() for i in ids],
                                  sum(len(v) for v in per_model.values()))}
        total = args.target

    accepted_per_model = {m: 0 for m in per_model}
    accepted = n_cases = n_requests = 0

    def next_job():
        if quota is None:
            return queues["*"].pop() if queues["*"] else None
        pend = [m for m in quota if accepted_per_model[m] < quota[m]]
        pend = [m for m in pend if queues[m]]
        if not pend:
            return None
        m = max(pend, key=lambda x: quota[x] - accepted_per_model[x])
        return (m, queues[m].pop())

    while accepted < total:
        job = next_job()
        if job is None:
            break
        gen_model, case_id = job
        n_cases += 1
        scenario = render_scenario(gts[(gen_model, case_id)]["extractions"])
        messages = [
            {"role": "system", "content": render("annotate_case", "system")},
            {"role": "user", "content": render("annotate_case", "user",
                                               scenario=scenario,
                                               text=texts[(gen_model, case_id)])},
        ]
        jobs = [(m, i) for m in args.models for i in range(1, args.per_model + 1)]
        log(f"case\t{case_id}\tgenerador={gen_model}\tjobs={len(jobs)}")
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(
                lambda j: (j[0], j[1],
                           ask_with_retries(j[0], messages, args.temperature,
                                            args.max_tokens, args.attempts, log)),
                jobs))
        n_requests += len(results)
        for model, rep, v in results:
            f_requests.write(json.dumps({"case_id": case_id, "generador": gen_model,
                                         "repeticion": rep, **v}, ensure_ascii=False) + "\n")
        votes = [v for _, _, v in results]
        record = {"case_id": case_id, "generador": gen_model,
                  "scenario": gts[(gen_model, case_id)]["extractions"],
                  "text": texts[(gen_model, case_id)],
                  "votos": [{"model": m, "rep": r, "apto": v["apto"],
                             "campos_graves": v["campos_graves"], "motivo": v["motivo"],
                             "error": v["error"], "finish_reason": v["finish_reason"]}
                            for m, r, v in results]}
        if all(v["apto"] is True for v in votes):
            accepted += 1
            accepted_per_model[gen_model] += 1
            f_ok.write(json.dumps(record, ensure_ascii=False) + "\n")
            log(f"accepted\t{case_id}\t{gen_model}\ttotal={accepted}/{total}"
                f"\tmodelo={accepted_per_model[gen_model]}/"
                f"{quota[gen_model] if quota else '-'}")
            print(f"[{accepted}/{total}] {case_id} ({gen_model}) "
                  f"{len(votes)}/{len(votes)} apto  "
                  f"{accepted_per_model[gen_model]}/{quota[gen_model] if quota else '-'}",
                  flush=True)
        else:
            f_ko.write(json.dumps(record, ensure_ascii=False) + "\n")
            log(f"rejected\t{case_id}\t{gen_model}")
        f_requests.flush(); f_ok.flush(); f_ko.flush()

    f_requests.close(); f_ok.close(); f_ko.close()
    log(f"end\tcasos={n_cases}\taceptados={accepted}\trechazados={n_cases - accepted}"
        f"\tpeticiones={n_requests}\tpor_generador={accepted_per_model}")
    log_file.close()
    print(f"\ncasos: {n_cases} | aceptados: {accepted} | rechazados: {n_cases - accepted}"
          f" | peticiones: {n_requests}")
    print(f"por generador: {accepted_per_model}")
    print(f"salida: {out}")


if __name__ == "__main__":
    main()
