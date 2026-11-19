# tools/ — generador

Scripts de análisis y medida. No forman parte del pipeline de generación.

| Script | Para qué |
|---|---|
| `dump_prompt_golden.py` | Regenera `tests/fixtures/prompt_golden.json` tras cambiar un prompt. Base de la regresión de plantillas. |
| `label_cases.py` | Anota casos de un corpus con un panel de LLMs (una petición por caso y modelo). Produce el gold del panel que consume `judge_eval --gold small`. |
| `build_judge_bench.py` | Construye el banco ciego (`cases.jsonl`, `batches/`) a partir de los `conversations.jsonl` de una campaña. |
| `judge_eval.py` | Mide un juez contra el gold del panel o contra el banco grande. Acepta `--votes`, `--prompt-dir` y `--annotators`. |
| `analyse_smoke.py` | Desglosa los modos de fallo de una campaña por modelo, leyendo sus `raw/`. |

## Rutas

Ningún script lleva rutas absolutas. Los datos de evaluación viven fuera del repo y se
indican con `--eval-dir`; el valor por defecto es `<repo>/eval`, que está gitignorado.
El endpoint del juez se pasa con `--base-url`; el del panel, con `--base-url` y
`--key-file`.

```bash
python tools/judge_eval.py --judge qwen3.5-9b-awq --base-url http://host:9292 \
  --eval-dir /ruta/al/archivo/eval --tag prueba
```

## Retirados

`judge_bench.py` y `calibrate_judge.py` se eliminaron: el primero quedó cubierto por
`judge_eval` (que ahora soporta `--annotators` y `--dump-verdicts`); el segundo dependía
de etiquetas del gold1k antiguo, fuera del repo.
