# Tareas pendientes de ejecución (handoff)

Estado a 20/10/2026. Todo lo documentado aquí tiene código y diseño listos; falta
ejecución (y en un caso, un rebuild de imagen docker).

---

## 1. Completar el dataset-10K con los 2 modelos excluidos (cierra G3)

- **Qué**: `gen_agentica-deepcoder-14b` y `gen_muse-glimmer-30b`, 1000 quejas cada uno
  (~17.6h de GPU en total: ETA 10.15h y 7.43h según `dataset-10k_exploration.json`).
- **Bloqueo encontrado**: la imagen docker `miurba-generator` está desactualizada — su
  `cli.py` no acepta `-n/--model` (la CLI cambió en main). **Paso previo obligatorio**:
  `docker build -t miurba-generator miurba-generator/` (aprobado por el usuario).
- **Cómo**: el script ya tiene soporte `--models` (añadido en housekeeping):

  ```bash
  cd /backup/pc1/data/miUrba2/workbench/document_experiments/mi-urba
  setsid nohup python3 eval/experiments/datasets/generate_10k_dataset.py \
      --models gen_agentica-deepcoder-14b,gen_muse-glimmer-30b \
      > /tmp/10k_complete.log 2>&1 &
  ```

- Requiere `LLAMA_GATEWAY_API_KEY` en el `.env` de la raíz del repo (ya copiado desde
  `miUrba2/mi-urba/.env`); el gateway es `192.168.10.11:9292` (llama-swap).
- `--resume` es automático (cuenta líneas en `raw/complaints.jsonl` por modelo).
- Verificado antes del corte: warmup de ambos modelos OK (56s / 34s).

## 2. E7 — juez de fidelidad: pasar del test al run completo (cierra G2)

- **Test ejecutado (20/10)**: dataset-5k, 10 quejas/modelo, 9 modelos con datos
  (`glm-4.7-flash` sin GT en el corpus). Resultados en
  `results/fidelidad/fidelity_summary.json` — phi-4 el más fiel (alucinación 0.0,
  urgencia 0.8); qwen3-0.6b el peor (completeness 0.53).
- **Ajustes hechos tras el test**: el prompt del juez ya no cuenta elaboraciones
  narrativas como invenciones; `completeness` cliñado por `gt_index`; `max_tokens` 3000;
  paralelismo `--workers 4` (~2.5s/juicio efectivo); resume automático.
- **Pendiente**:
  1. Test pequeño de consistencia del nuevo prompt (relanzar 10/modelo sobre dataset-5k
     comparando con la primera tanda; se mezclan 2 versiones del prompt en los JSONL
     actuales — valorar limpiar `results/fidelidad/` y regenerar).
  2. Run completo: corpus primario `results/diversity/gen_*/` (81 modelos × 20, escenarios
     idénticos → comparación limpia y correlación con fingerprints de E3):
     ```bash
     python3 eval/experiments/fidelidad_generador/judge_fidelity.py \
         --dataset eval/results/diversity --n-per-model 20
     ```
  3. Controles del diseño (`experiments/fidelidad_generador/README.md`): control negativo,
     auto-consistencia 2 pasadas, validación humana de 30 juicios (el usuario actúa de
     validador).
  4. Triangulación con E4: correlacionar completeness del juez con recall del extractor
     por modelo.
- Requiere el juez `qwen3.8-27b-awq` cargado en el gateway.

## 3. Otros pendientes menores

- **Actualizar G3 en `README.md`** cuando se ejecute 1: hoy dice "10K parcial por
  warmup timeouts" — la causa real fue el corte por ETA (documentado en
  `generate_10k_dataset.py` L52 y `dataset-10k_exploration.json`).
- **`glm-4.7-flash` en dataset-5k**: directorio sin `ground_truth.jsonl` — regenerar o
  documentar como hueco del corpus.
- **README de `eval/` como portada definitiva**: el actual ya tiene el hilo conductor y
  los gaps; pulirlo cuando 1 y 2 estén cerrados (G2/G3 pasarán de "gap" a "experimento").
- **Sincronizar los otros checkouts** (`/backup/pc1/data/miUrba2/mi-urba/`,
  `/home/dani/xppl/`) — siguen en estado pre-housekeeping; marcarlos como congelados
  para evitar divergencia.
