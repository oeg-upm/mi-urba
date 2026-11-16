#!/usr/bin/env python
"""
miUrba — Generador de imágenes con FLUX.2-dev 32B (NF4) — CLI auxiliar.

Este script es una alternativa standalone a la generación de imágenes
integrada en el pipeline (`cli.py generate` con `image_generation.enabled`).
Reutiliza el mismo backend (`src.imagegen.FluxImageGenerator`) y los mismos
prompts (`src.prompts.build_image_prompts`), derivados del scenario.

Modos:
  --test                 una imagen de prueba con prompt fijo
  --metadata <file>      genera imágenes para cada registro de metadata.jsonl
                         (usa los `image_prompts` guardados por el generador,
                          o los reconstruye a partir del scenario)
  (sin flags)            12 incidencias de demostración con prompts fijos

Requiere GPU y dependencias extra: ver requirements-images.txt.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.imagegen import FluxImageGenerator
from src.prompts import build_image_prompts


def main():
    parser = argparse.ArgumentParser(description="miUrba FLUX.2 image generator (multi-GPU)")
    parser.add_argument("--test", action="store_true", help="Test rápido")
    parser.add_argument("--metadata", type=str, help="Path a metadata.jsonl")
    parser.add_argument("--output", type=str, default="data/images", help="Directorio salida")
    parser.add_argument("--steps", type=int, default=28, help="Inference steps")
    parser.add_argument("--guidance", type=float, default=4.0, help="Guidance scale")
    parser.add_argument("--size", type=int, default=1024, help="Tamaño imagen")
    parser.add_argument("--limit", type=int, default=0, help="Limitar registros (0=todos)")
    args = parser.parse_args()

    gen = FluxImageGenerator(size=args.size, steps=args.steps, guidance=args.guidance)
    gen.load()

    if args.test:
        print("=== Test ===")
        prompt = ("Documentary photograph of a ruptured galvanized steel water pipe "
                  "leaking water in an underground parking garage, concrete walls and floor, "
                  "fluorescent ceiling light, water pooling on the floor, "
                  "professional inspection photo, no people")
        t0 = time.time()
        os.makedirs(args.output, exist_ok=True)
        gen.generate(prompt, 42, f"{args.output}/test_flux2_dev.png")
        print(f"Generada en {time.time()-t0:.1f}s -> {args.output}/test_flux2_dev.png")
        return

    if args.metadata:
        print(f"=== Generando desde {args.metadata} ===")
        os.makedirs(args.output, exist_ok=True)
        records = []
        with open(args.metadata) as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        if args.limit > 0:
            records = records[:args.limit]
        total = 0
        for rec in records:
            rid = rec["id"]
            # Prefer prompts saved by the generator; otherwise rebuild from scenario
            prompts = rec.get("image_prompts") or build_image_prompts(
                rec.get("problems", []), rec.get("community", {}), rec.get("num_pics", 0)
            )
            for i, prompt in enumerate(prompts, 1):
                out_path = f"{args.output}/{rid}_{i}.png"
                if os.path.exists(out_path):
                    print(f"  {rid}_{i} existe, saltando")
                    continue
                print(f"  {rid}_{i}...", end=" ", flush=True)
                t0 = time.time()
                seed = sum((k + 1) * ord(c) for k, c in enumerate(rid)) * 1000 + i
                gen.generate(prompt, seed % (2**32), out_path)
                print(f"{time.time()-t0:.0f}s")
                total += 1
        print(f"\n=== {total} imágenes en {args.output} ===")
        return

    # 12 incidencias de demostración
    print("=== 12 incidencias ===")
    incidencias = [
        ("01_fuga_tuberia", "A ruptured galvanized steel water pipe leaking water in an underground parking garage, concrete walls and floor, fluorescent ceiling light, water pooling on the floor, professional inspection photo, no people"),
        ("02_ascensor_falla", "A broken elevator control panel in an apartment building hallway, exposed wires, damaged buttons, yellow caution tape, fluorescent lighting, no people"),
        ("03_cableado_electrico", "Deteriorated electrical wiring in a building entrance corridor, exposed cables, cracked conduit, old switchbox, concrete walls, no people"),
        ("04_contenedores_desbordados", "Overflowing communal garbage containers in a residential courtyard, trash spilling on the ground, brick building facade, daylight, no people"),
        ("05_humedad_pared", "Extreme close-up of water damage and black mold on an interior wall, peeling paint, damp stains, plaster deterioration, no people"),
        ("06_cerradura_forzada", "Forced entry lock damage on a modern apartment entrance door, scratched metal, bent latch, broken frame, no people"),
        ("07_jardin_abandonado", "An abandoned communal garden in a residential complex, overgrown weeds, dead bushes, cracked paths, building facade in background, no people"),
        ("08_pared_derribada", "A demolished partition wall between apartments, exposed rebar, rubble on floor, dust, construction debris, no people"),
        ("09_balsa_azotea", "Large stagnant water pool on a flat building rooftop, green algae, cracked waterproof membrane, parapet wall, overcast sky, no people"),
        ("10_caldera_comunitaria", "An industrial communal boiler room with failing equipment, rusted pipes, pressure gauges, metal casing, concrete floor, no people"),
        ("11_parking_agrietado", "Severely cracked and sunken underground parking floor, concrete damage, structural cracks, painted parking lines, no people"),
        ("12_gotera_techo", "An active ceiling water leak in a residential interior, water stain on ceiling, dripping water, damaged plaster, furniture below, no people"),
    ]
    os.makedirs(args.output, exist_ok=True)
    total_time = 0
    for idx, (name, prompt) in enumerate(incidencias):
        print(f"[{idx+1}/12] {name}...", end=" ", flush=True)
        t0 = time.time()
        gen.generate(prompt, 42 + idx, f"{args.output}/{name}.png")
        elapsed = time.time() - t0
        total_time += elapsed
        print(f"{elapsed:.0f}s")
    print(f"\n=== 12 imágenes en {total_time:.0f}s ({total_time/12:.0f}s/img) ===")


if __name__ == "__main__":
    main()