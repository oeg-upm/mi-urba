# E7 · Fidelidad del generador al escenario (juez LLM)

Cierra el gap **G2** del `../../README.md`: ninguna métrica existente responde
directamente "¿el texto que escribe el generador refleja el escenario con el que se
generó?". E4 lo mide al revés (extractor→GT, contaminado por los errores del extractor)
y E5 usa esa medida como proxy. Aquí un **juez LLM** compara texto y GT directamente.

## Pregunta

> ¿Qué generadores escriben quejas fieles a su escenario, y en qué dimensiones fallan?

## Diseño

### Juez

`qwen3.8-27b-awq` servido por llama-swap en `http://192.168.10.11:9292`
(modelo distinto de todos los generadores evaluados; tamaño medio, no awq de generación
para evitar sesgo de linaje con los evaluados).

### Muestra

- **Primario**: `results/dataset-10k/` — 8 modelos × 100 quejas (muestreo aleatorio con
  seed) = 800 juicios. Es el corpus de la selección iter 2.
- **Opcional**: `results/diversity/gen_*/` (20/modelo) para extender a los 81 modelos.

### Dimensiones de juicio (por queja, JSON estructurado)

El juez recibe el GT completo (todos los problemas del escenario) y el texto, y devuelve:

| Dimensión | Definición |
|---|---|
| `completeness` | para cada problema del GT: ¿aparece en el texto? (por-problema, no agregado) |
| `faithfulness` | ¿el texto inventa problemas/riesgos/elementos ajenos al GT? (lista de invenciones) |
| `urgency_match` | ¿el texto transmite la urgencia GT de cada problema? (exact/not/exaggerated) |
| `risks_match` | ¿los riesgos GT aparecen sin añadir otros? |
| `naturalness` | 1-5: ¿parece una queja vecinal real? |

### Métricas agregadas por generador

- `completeness_rate` = problemas GT presentes / problemas GT totales (≈ recall sintético)
- `hallucination_rate` = quejas con ≥1 invención / quejas totales
- `urgency_accuracy`, `risks_accuracy` (según veredicto del juez)
- `naturalness_mean`
- **Score de fidelidad** = media z-scored de las 4 primeras (naturalness se reporta aparte:
  mide calidad, no fidelidad)

### Controles de validez

1. **Control negativo**: para 50 quejas, se elimina quirúrgicamente un problema del GT
   del texto (segunda pasada del propio generador con consigna de borrado) → el juez
   debe detectar el fallo. Si `completeness_rate` no cae, el juez es sordo.
2. **Auto-consistencia**: 2 pasadas del juez (temp 0.3) sobre 100 quejas → acuerdo
   por dimensión. Si κ/ρ bajo, la dimensión no es fiable y se documenta.
3. **Validación humana (G4, muestra pequeña)**: 30 juicios aleatorios para revisión del
   usuario contra el veredicto del juez → accuracy humana del juez.

### Triangulación con E4

Correlacionar `completeness_rate` del juez con el recall del extractor (E4) por modelo:
si un generador escribe fiel pero el extractor recupera mal, la culpa es del extractor;
si el juez ya lo puntúa bajo, la culpa es del texto. Esto separa por fin las dos
mitades del canal generador→texto→extractor.

## Entregables

- `results/fidelidad/judgments_<modelo>.jsonl` (un juicio por línea, con cruda del juez)
- `results/fidelidad/fidelity_summary.json` (métricas por generador + controles)
- Informe corto en este README (sección Resultados)

## Ejecución

```bash
python3 eval/experiments/fidelidad_generador/judge_fidelity.py \
    --dataset eval/results/dataset-10k --n-per-model 100
```

Requiere el juez cargado en el gateway (192.168.10.11:9292) y `LLAMA_GATEWAY_API_KEY`
en el `.env` del repo o en el entorno.

## Resultados

_(pendiente de primera ejecución)_
