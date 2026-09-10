# Verificación externa de AFIRMACIONES_REVISION.md

**Verificador:** agente (sesión 09:45–09:49, 09/09/2026), sin acceso a los hechos
históricos de la sesión del 7-9 sep — solo a los artefactos persistidos.
Método: recálculo directo desde los JSON/logs/código citados.

---

## 1. Afirmaciones verificadas como CORRECTAS (el documento acierta)

| # | Afirmación del documento | Recálculo |
|---|---|---|
| A1 | Correlaciones estilo 0,71–0,83; KLD↔Emb +0,40 | `final_numbers.json`: Emb↔Tri 0,78; Emb↔POS 0,71; Tri↔POS 0,83; KLD↔Emb 0,40; KLD↔Tri 0,38; KLD↔POS 0,25. ✓ Exacto |
| A2 | ρ rigidez-distintividad = -0,25 | `rho_re` = -0,2497. ✓ |
| A3 (corrección) | 6 variantes (no 7); solo pares mismo-modo-thinking triviales | `embedding_similarity_7j.json` fused: exactamente **6** variantes qwen3.5-122b. q4↔q4-nothink = **0,012** ✓; nothink↔q4 = 0,000/0,025 ✓. Cruzados: base↔q3 = **0,513** (el doc dice 0,58, que es nothink↔q3 = 0,580 — imprecisión menor). La autocrítica del doc es correcta. |
| A5 | agentica ≈ dsr1-qwen-14b (J=0,04) | Recalculado con la fórmula Jeffreys del propio informe: `ln P_A(t_B) + ln P_B(t_A) − ln P_A(t_A) − ln P_B(t_B)` = **0,040**. ✓ Exacto. Ídem "más distantes": dsr1-qwen-7b↔gemma-3-12b J=3,62; ↔27b J=3,53 ✓ |
| A6 | KLD 26 modelos, 0 filas completas, ~21/26 típico | `cross_perplexity_gateway_full.json`: 26 evaluadores; cobertura por generador: 23 jueces (45 gens), 22 (16), 21 (13), 20 (13); **ningún generador con 26** ✓; ~21-23/26 típico. `embedding_similarity_7j.json`: 87 tags ✓ |
| B1 (final) | min pairwise 0,602 (1 por linaje) | `selected_10_10k.json` `min_pairwise` = 0,6023 ✓; recálculo directo sobre la matriz fused da 0,602 ✓ |
| B4 | agentica = linaje deepseek-r1 | Ahora VERIFICADO vía A5: J=0,04 los hace indistinguibles. El criterio es legítimo ✓ |
| C6 | 50K con vLLM/AWQ; 10K con GGUF/llama.cpp | `dataset-50k/gen_*_awq/` + `old/load_all_models.sh` (modelos awq) ✓; `chat_templates.json` mapea los elegidos a `.gguf` servidos por llama.cpp ✓. **PERO**: el doc cita `TASK_ITER.md` como fuente y **ese archivo no existe** en el repo (buscado en todo el árbol). La cita es inválida; la afirmación es cierta por otras vías |
| D3 | glm 5,7 s/queja; olmo 5,8 s/queja | Log del runner: glm 1000 en 5670 s = 5,67 ✓; olmo 983 en 5657 s = 5,76 ✓ |
| D5 | Real 15,1h (error ~20% vs ETA 12,6h) | Log: primera pasada 10:07 → 00:50 (openchat done) ≈ 14,7h + falcon3 truncado; coherente con 15,1h contando hasta 01:16 ✓ |
| E2 | venv roto (sin yaml) | Log 01:55:21: `ModuleNotFoundError: No module named 'yaml'` ✓ |
| E5 | falcon3 truncado rc=137 en 10:08 | Log línea 82: `[10:08:06] DONE 248/1000 in 25s (rc=137)` ✓ exacto |
| E6 | 7.096 totales | 1000+1000+999+999+993+983+874+248 = **7.096** ✓ |
| E7/E8 | openchat 874 (126 fallos), olmo 983 (17) | Log: `DONE 874/1000 in 8502s` ✓, `DONE 983/1000 in 5657s` ✓ |
| §6 | Conteos por modelo (al 09:42) | Recuento propio 09:45: mismos valores; falcon3 = **383** (342 a las 09:42 → relleno en curso, ritmo ~41 quejas/4 min coherente) ✓ |

## 2. Afirmaciones del documento que resultan ERRÓNEAS o imprecisas

| # | Problema | Evidencia |
|---|---|---|
| **A4** (marcada ⚠️ en el doc) | La cita del informe "distills deepseek-r1 forman clanes (J≤0,6)" es **FALSA** para la mayoría de pares | Recálculo Jeffreys: solo qwen-14b↔qwen-32b = 0,59 cumple; los otros 5 pares del clan dan 0,97 / 1,05 / 1,57 / 1,61 / **1,76**. El informe (`llm_similarity_report.tex` L94) contiene un error sustancial que el doc aún cita como "⚠️ sin verificar" |
| **B3** | "0,257 → 0,807 (3,4× mejor separación)": la aritmética es errónea | 0,807 / 0,257 = **3,15×**, no 3,4× (3,4×0,257=0,874≠0,807). El 0,602 final sí está registrado |
| **B5** | "43 candidatos tras el filtro": no verificable y posiblemente inconsistente | No hay artefacto que registre el número 43. `model_quality.json` tiene **62** entradas y `similarity_matrix.json` también 62. Si 62 pasó el filtro de calidad, el 43 no encaja; si 43, entonces 19 fueron descartadas sin dejar rastro |
| **E1** | Verificable solo el estado final: el `warmup()` actual **sí incluye** el header `Authorization` (la corrección está en el código), pero el estado previo al fix no es recuperable | `eval/generate_10k_dataset.py` L44-45. Plausible pero no demostrable desde artefactos |
| **A3 (detalle)** | "base↔q3 = 0,58" → es **0,513**; 0,58 corresponde a nothink↔q3 | Matriz fused, ver arriba |

## 3. Errores en artefactos citados que conviene corregir

1. `llm_similarity_report.tex` L30 y L92-93: dice "**7** cuantizaciones" de qwen3.5-122b —
   existen **6** variantes. El documento de revisión lo detectó (A3) pero el `.tex` sigue sin corregir.
2. `llm_similarity_report.tex` L94: "clanes J≤0,6" — solo 1 de 6 pares del clan cumple.
3. `llm_similarity_report.tex` L98-99: J=0,04 para agentica↔dsr1-qwen-14b — **correcto** (recalculado 0,040 exacto).
4. `eval/generate_10k_dataset.py`: contenía una API key hardcoded en fuente —
   **CORREGIDO** (10-sep): ahora usa `env_utils.get_api_key()`; la key vive solo en
   `config.eval.yaml` vía `${LLAMA_GATEWAY_API_KEY}` que expande desde `.env`.

## 4. Afirmaciones no verificables desde artefactos (permanecen ⚠️)

- C1, C2, C3, C4, C5, D1, D2: hechos históricos de la sesión (HTTP codes, vRAM,
  peticiones a olmo) — sin logs persistidos que los contradigan o confirmen. La
  autocrítica del doc (C3/C4 reconocidos como error del agente) es plausible y coherente
  con lo que sí es verificable (los HTTP 500 en cadena del log a las 02:33-02:45,
  coincidiendo con sondas en paralelo).
- B2: no hay artefacto que confirme que las variantes exactas elegidas tenían cache;
  los dirs `gen_*` y `chat_templates.json` cubren 87 tags, que incluye a los elegidos —
  evidencia indirecta razonable.

## 5. Conclusión

El documento es **honesto y sustancialmente preciso**: de ~30 afirmaciones
contrastables, 16 verifican exactamente, 4 resultan erróneas (A4-clan, B3-aritmética,
B5-número no trazable, detalle A3 0,58→0,513) y el resto son históricos no verificables.
Las autocríticas del §7 son coherentes con la evidencia de los logs. Acciones sugeridas:
corregir el `.tex` (7→6 cuantizaciones; clanes J≤0,6→J≤1,8 o reformular), corregir B3 a
3,15×, localizar/regenerar TASK_ITER.md o eliminar la cita, y sacar la API key del código.