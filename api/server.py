#!/usr/bin/env python3
"""
miUrba API — servidor FastAPI con endpoints estilo OpenAI.

Expone las dos tareas del proyecto (generación y extracción) via REST:
  GET  /v1/models              lista modelos disponibles (formato OpenAI)
  GET  /v1/health              health check
  POST /v1/generate            genera quejas sintéticas (LLM)
  POST /v1/extract             extrae estructura de quejas (VLM)
  POST /v1/pipeline            ejecuta generate + extract en un solo call

Uso:
  uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any

import os

import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ── Paths ────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent
GENERATOR_DIR = REPO_ROOT / "miurba-generator"
EXTRACTOR_DIR = REPO_ROOT / "miurba-extractor"
DEFAULT_CONFIG = REPO_ROOT / os.environ.get("MIURBA_CONFIG", "config.yaml")

logger = logging.getLogger("miurba.api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")

app = FastAPI(
    title="miUrba API",
    description="Generación de quejas sintéticas y extracción estructurada con VLM",
    version="1.0.0",
)

# ── Config loading ───────────────────────────────────────────────────────────


def load_config(config_path: str | None = None) -> dict[str, Any]:
    path = Path(config_path) if config_path else DEFAULT_CONFIG
    if not path.exists():
        raise HTTPException(status_code=400, detail=f"Config not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)


def get_models(config: dict) -> dict[str, dict]:
    """Return model configs in a flat dict keyed by model name."""
    return config.get("models", {})


# ── Request / Response models ────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    model: str = "kimi-k2.6"
    n: int = Field(default=3, ge=1, le=100)
    config: str | None = None  # path to config file override


class ExtractRequest(BaseModel):
    model: str | None = None  # VLM model; default: annotation.vision_model or annotation.model
    complaints: list[dict[str, Any]]  # array of complaint objects (conforme a complaint_schema.json)
    images: dict[str, str] | None = None  # {filename: base64_data} for photos
    config: str | None = None


class PipelineRequest(BaseModel):
    generate_model: str = "kimi-k2.6"
    extract_model: str | None = None
    n: int = Field(default=3, ge=1, le=100)
    config: str | None = None


class GenerationResponse(BaseModel):
    object: str = "generation"
    id: str
    model: str
    complaints: list[dict[str, Any]]
    ground_truth: list[dict[str, Any]]
    metadata: list[dict[str, Any]]
    stats: dict[str, Any]


class ExtractionResponse(BaseModel):
    object: str = "extraction"
    id: str
    model: str
    extractions: list[dict[str, Any]]


class PipelineResponse(BaseModel):
    object: str = "pipeline"
    id: str
    generate_model: str
    extract_model: str
    complaints: list[dict[str, Any]]
    ground_truth: list[dict[str, Any]]
    metadata: list[dict[str, Any]]
    extractions: list[dict[str, Any]]


class ModelObject(BaseModel):
    id: str
    object: str = "model"
    owned_by: str = "miurba"
    capabilities: dict[str, bool] = {}


class ModelsResponse(BaseModel):
    object: str = "list"
    data: list[ModelObject]


class HealthResponse(BaseModel):
    status: str
    version: str
    generator: bool
    extractor: bool


# ── Subprocess helpers ───────────────────────────────────────────────────────

def run_generator(model: str, n: int, config_path: str, output_dir: str) -> tuple[dict, list, list]:
    """Run the generator CLI as subprocess, return (stats, complaints, ground_truth, metadata)."""
    cmd = [
        sys.executable, str(GENERATOR_DIR / "cli.py"),
        "generate",
        "-n", str(n),
        "--model", model,
        "--config", config_path,
        "--output-dir", output_dir,
        "--no-resume",
    ]
    logger.info(f"Running generator: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise HTTPException(status_code=502, detail=f"Generator failed: {result.stderr[-500:]}")

    complaints = []
    raw_dir = Path(output_dir) / "raw"
    cfile = raw_dir / "complaints.jsonl"
    gfile = raw_dir / "ground_truth.jsonl"
    mfile = raw_dir / "metadata.jsonl"
    complaints = []
    ground_truth = []
    metadata = []
    if cfile.exists():
        with open(cfile) as f:
            complaints = [json.loads(l) for l in f if l.strip()]
    if gfile.exists():
        with open(gfile) as f:
            ground_truth = [json.loads(l) for l in f if l.strip()]
    if mfile.exists():
        with open(mfile) as f:
            metadata = [json.loads(l) for l in f if l.strip()]

    # Parse stats from stdout
    stats = {"generated": len(complaints)}
    return stats, complaints, ground_truth, metadata


def run_extractor(
    model: str | None,
    config_path: str,
    complaints: list[dict],
    images: dict[str, str] | None,
    work_dir: str,
) -> list[dict]:
    """Run the extractor CLI as subprocess, return extractions list."""
    raw_dir = Path(work_dir) / "raw"
    images_dir = Path(work_dir) / "images"
    annotated_dir = Path(work_dir) / "annotated"
    raw_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir.mkdir(parents=True, exist_ok=True)

    # Write complaints to JSONL
    with open(raw_dir / "complaints.jsonl", "w", encoding="utf-8") as f:
        for c in complaints:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    # Write images (base64 decoded)
    if images:
        import base64
        for filename, b64data in images.items():
            img_path = images_dir / filename
            img_path.parent.mkdir(parents=True, exist_ok=True)
            with open(img_path, "wb") as f:
                f.write(base64.b64decode(b64data))

    cmd = [
        sys.executable, str(EXTRACTOR_DIR / "cli.py"),
        "extract",
        "--config", config_path,
        "--raw-dir", str(raw_dir),
        "--images-dir", str(images_dir),
        "--annotated-dir", str(annotated_dir),
        "--no-resume",
    ]
    if model:
        cmd += ["--model", model]

    logger.info(f"Running extractor: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise HTTPException(status_code=502, detail=f"Extractor failed: {result.stderr[-500:]}")

    ext_file = annotated_dir / "extractions.jsonl"
    extractions = []
    if ext_file.exists():
        with open(ext_file) as f:
            extractions = [json.loads(l) for l in f if l.strip()]
    return extractions


def resolve_config(config_path: str | None) -> str:
    """Resolve config path: absolute, or relative to repo root."""
    if config_path is None:
        return str(DEFAULT_CONFIG)
    p = Path(config_path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    if not p.exists():
        raise HTTPException(status_code=400, detail=f"Config not found: {p}")
    return str(p)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/v1/health", response_model=HealthResponse, tags=["meta"])
async def health():
    """Health check."""
    return HealthResponse(
        status="ok",
        version="1.0.0",
        generator=(GENERATOR_DIR / "cli.py").exists(),
        extractor=(EXTRACTOR_DIR / "cli.py").exists(),
    )


@app.get("/v1/models", response_model=ModelsResponse, tags=["meta"])
async def list_models(config: str | None = None):
    """List available models (OpenAI-style format)."""
    cfg = load_config(resolve_config(config))
    annotation = cfg.get("annotation", {}) or {}
    vision_model = annotation.get("vision_model") or ""
    models_cfg = get_models(cfg)
    data = []
    for name, mc in models_cfg.items():
        data.append(ModelObject(
            id=name,
            owned_by=mc.get("endpoint", "miurba"),
            capabilities={
                "vision": name in vision_model or
                          (annotation.get("model") == name),
            },
        ))
    return ModelsResponse(data=data)


@app.post("/v1/generate", response_model=GenerationResponse, tags=["generation"])
async def generate(req: GenerateRequest):
    """
    Generate synthetic complaints using an LLM.

    Calls the miurba-generator pipeline to produce `n` complaint records
    with the specified model. Returns complaints + ground_truth + metadata.
    """
    config_path = resolve_config(req.config)
    cfg = load_config(config_path)
    if req.model not in cfg.get("models", {}):
        raise HTTPException(status_code=400, detail=f"Unknown model: {req.model}")

    gen_id = f"gen_{uuid.uuid4().hex[:12]}"
    with tempfile.TemporaryDirectory() as tmpdir:
        stats, complaints, ground_truth, metadata = run_generator(
            model=req.model,
            n=req.n,
            config_path=config_path,
            output_dir=tmpdir,
        )
        return GenerationResponse(
            id=gen_id,
            model=req.model,
            complaints=complaints,
            ground_truth=ground_truth,
            metadata=metadata,
            stats=stats,
        )


@app.post("/v1/extract", response_model=ExtractionResponse, tags=["extraction"])
async def extract(req: ExtractRequest):
    """
    Extract structured data from complaints using a VLM.

    Accepts an array of complaint objects (conforming to complaint_schema.json)
    and optionally base64-encoded images. Returns structured extractions with
    image descriptions, conforming to extraction_schema.json.
    """
    if not req.complaints:
        raise HTTPException(status_code=400, detail="No complaints provided")

    config_path = resolve_config(req.config)
    cfg = load_config(config_path)

    # Resolve VLM model
    model = req.model
    if model and model not in cfg.get("models", {}):
        raise HTTPException(status_code=400, detail=f"Unknown model: {model}")

    ext_id = f"ext_{uuid.uuid4().hex[:12]}"
    with tempfile.TemporaryDirectory() as tmpdir:
        extractions = run_extractor(
            model=model,
            config_path=config_path,
            complaints=req.complaints,
            images=req.images,
            work_dir=tmpdir,
        )
        return ExtractionResponse(
            id=ext_id,
            model=model or cfg.get("annotation", {}).get("vision_model") or cfg.get("annotation", {}).get("model", "unknown"),
            extractions=extractions,
        )


@app.post("/v1/pipeline", response_model=PipelineResponse, tags=["pipeline"])
async def pipeline(req: PipelineRequest):
    """
    Run the full pipeline: generate complaints → extract structure.

    Generates `n` synthetic complaints with the specified LLM model, then
    immediately extracts structured data from them with a VLM. Returns all
    intermediate and final outputs in one response.
    """
    config_path = resolve_config(req.config)
    cfg = load_config(config_path)
    if req.generate_model not in cfg.get("models", {}):
        raise HTTPException(status_code=400, detail=f"Unknown generate model: {req.generate_model}")

    pipe_id = f"pipe_{uuid.uuid4().hex[:12]}"

    with tempfile.TemporaryDirectory() as tmpdir:
        # Phase 1: Generate
        stats, complaints, ground_truth, metadata = run_generator(
            model=req.generate_model,
            n=req.n,
            config_path=config_path,
            output_dir=tmpdir,
        )

        # Phase 2: Extract
        extract_model = req.extract_model
        extractions = run_extractor(
            model=extract_model,
            config_path=config_path,
            complaints=complaints,
            images=None,  # no images in pipeline mode (FLUX disabled)
            work_dir=tmpdir,
        )

        return PipelineResponse(
            id=pipe_id,
            generate_model=req.generate_model,
            extract_model=extract_model or cfg.get("annotation", {}).get("vision_model") or cfg.get("annotation", {}).get("model", "unknown"),
            complaints=complaints,
            ground_truth=ground_truth,
            metadata=metadata,
            extractions=extractions,
        )


@app.get("/", tags=["meta"])
async def root():
    """API root — redirect to docs."""
    return {
        "name": "miUrba API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": ["/v1/health", "/v1/models", "/v1/generate", "/v1/extract", "/v1/pipeline"],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.server:app", host="0.0.0.0", port=8000, reload=True)