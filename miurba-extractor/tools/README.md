# tools/ — extractor

| Script | Para qué |
|---|---|
| `dump_prompt_golden.py` | Regenera `tests/fixtures/prompt_golden.json` tras cambiar un prompt. Uso: `python tools/dump_prompt_golden.py [out.json]` |

Sin rutas absolutas ni variables de entorno: la ruta por defecto es el fixture del
propio subproyecto.
