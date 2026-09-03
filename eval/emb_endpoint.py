#!/usr/bin/env python3
"""Micro-endpoint de embeddings compatibles con /v1/embeddings (OpenAI schema).

Sirve modelos sentence-transformers (encoders clásicos tipo BERT/RoBERTa que
llama.cpp no soporta) con el mismo schema que llama-server, para que el
pipeline de jueces no cambie.
"""
import argparse
from typing import List

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import uvicorn

app = FastAPI()
MODEL = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class EmbedReq(BaseModel):
    input: List[str]


@app.post("/v1/embeddings")
def embeddings(req: EmbedReq):
    E = MODEL.encode(req.input, batch_size=16, normalize_embeddings=True,
                     convert_to_numpy=True)
    return {"object": "list", "data": [
        {"object": "embedding", "index": i, "embedding": e.tolist()}
        for i, e in enumerate(E)]}


@app.get("/health")
def health():
    return {"status": "ok", "model": getattr(MODEL, "_model_card_vars", {}) and True}


def serve(model_id: str, port: int):
    global MODEL
    print(f"loading {model_id} on {DEVICE}...", flush=True)
    MODEL = SentenceTransformer(model_id, device=DEVICE)
    print("loaded", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    import sys
    serve(sys.argv[1], sys.argv[2])