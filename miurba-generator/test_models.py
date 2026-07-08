#!/usr/bin/env python3
"""Test de modelos NeuralWatt con salida completa para evaluación.

Auxiliar del proyecto miurba-generator. Compara las respuestas de los
modelos configurados sobre un escenario fijo.

Uso:
    python test_models.py
"""

import json
import time
import sys
from pathlib import Path

import yaml

# Resolve paths relative to this file (miurba-generator/test_models.py)
HERE = Path(__file__).resolve().parent          # .../miurba-generator
REPO_ROOT = HERE.parent                          # .../mi-urba (config compartida)

sys.path.insert(0, str(HERE))                     # para que `src` sea miurba-generator/src

with open(REPO_ROOT / "config.yaml", encoding="utf-8") as f:
    base_config = yaml.safe_load(f)
with open(REPO_ROOT / "config.neuralwatt.yaml", encoding="utf-8") as f:
    nw_config = yaml.safe_load(f)
merged = {**base_config, **nw_config}

from src import config as cfg
cfg._CONFIG = merged
from src.config import get_config, get_model_params
from src.llm import LLMClient
from src.sampling import sample_scenario, sample_community, sample_channel, sample_persona, sample_modifier
from src.prompts import build_generation_prompt

config = get_config()

# Escenario fijo para comparación justa
scenario = sample_scenario()
community = sample_community()
channel = sample_channel()
persona = sample_persona()
modifier = sample_modifier()
num_pics = 1

print(f"Escenario: {scenario['category']}/{scenario['subcategory']} en {scenario['location_zone']}")
print(f"Comunidad: {community['name']}")
print(f"Canal: {channel}, Persona: {persona['id']}, Fotos: {num_pics}")
print()

models = list(config["models"].keys())
all_results = {}

for model_name in models:
    model_cfg = get_model_params(model_name, config)
    print(f"[{model_name}]...", end=" ", flush=True)

    client = LLMClient(
        base_url=model_cfg["base_url"],
        model=model_cfg["model"],
        temperature=model_cfg["temperature"],
        timeout=120,
        api_key=model_cfg.get("api_key"),
    )

    messages = build_generation_prompt(
        record_id="TEST", problems=[scenario], channel=channel,
        community=community, num_pics=num_pics,
        persona=persona, modifier=modifier,
    )

    t0 = time.time()
    try:
        response = client.chat(messages)
        data = client.extract_json(response)
        elapsed = time.time() - t0
        tokens = response.get("usage", {}).get("completion_tokens", 0)

        result = {
            "model": model_name,
            "time": round(elapsed, 1),
            "tokens": tokens,
            "user_text": data.get("user_text", ""),
            "image_descriptions": data.get("image_descriptions", []),
            "timestamp": data.get("timestamp", ""),
        }
        all_results[model_name] = result
        print(f"✅ {elapsed:.0f}s | {tokens} tok")
    except Exception as e:
        elapsed = time.time() - t0
        all_results[model_name] = {"model": model_name, "error": str(e)[:200], "time": round(elapsed, 1)}
        print(f"❌ {elapsed:.0f}s | {str(e)[:60]}")

# Guardar resultados completos
out_path = HERE / "test_results.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)

print(f"\nResultados guardados en {out_path}")
print(f"Modelos exitosos: {sum(1 for v in all_results.values() if 'user_text' in v)}/{len(all_results)}")