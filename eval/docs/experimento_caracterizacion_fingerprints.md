# Experimento de caracterización de generadores: fingerprints multidimensionales

> Fuentes: `eval/experiments/seleccion_modelos/embed_similarity.py`, `eval/experiments/seleccion_modelos/emb_endpoint.py`, `eval/experiments/seleccion_modelos/stylometry_analysis.py`,
> `eval/experiments/seleccion_modelos/rigidity_probe.py`, `eval/experiments/seleccion_modelos/final_analysis.py`, `eval/experiments/seleccion_modelos/analyze_cross_perplexity.py`.
> Resultados: `eval/results/diversity/{embedding_similarity,embedding_similarity_7j,
> stylometry_analysis,rigidity_probe,final_numbers}.json`, `mds_voces*.png`,
> `stylometry_mds_e1.png`, `fig_*.png`, `llm_similarity_report.tex/.pdf`.
> La familia 1 (cross-perplexity) tiene su propio documento:
> `familia1_cross_perplexity.md`.

---

## 1. La pregunta

> **¿En qué se diferencian realmente los 87 generadores de quejas, y cuánto?**

"Tener 87 generadores" solo tiene sentido si escriben de forma genuinamente distinta.
Este experimento mide esas diferencias con **cuatro familias de métricas independientes**
entre sí — por método, por recurso y por sesgo — de modo que lo en que **convergen**
es robusto y lo en que **divergen** es informativo. Además valida un control negativo:
las cuantizaciones del mismo modelo deberían parecerse entre sí casi trivialmente.

El corpus es común a las cuatro familias: los textos de `diversity/gen_<model>/`
(20 quejas por modelo, escenarios idénticos, prompt de producción iter 6).

## 2. Resultado principal (para orientar la lectura)

Las cuatro familias revelan **dos ejes ortogonales**:

- **Eje estilístico** ("cómo se escribe"): embeddings, trigramas y POS/dep convergen
  fuertemente — ρ de Spearman 0.71–0.83 entre sus matrices de distancias
  (`final_numbers.json`: Emb↔Tri 0.78, Emb↔POS 0.71, Tri↔POS 0.83).
- **Eje semántico** ("qué se dice"): solo KLD y embeddings lo capturan, y de forma
  parcial — ρ KLD↔Emb = +0.40; KLD↔Tri 0.38, KLD↔POS 0.25.

Y el control de rigidez: ρ rigidez vs distintividad = **−0.25** (`rho_re` = −0.2497).

Con líneas así, cada familia se lee como una lente distinta sobre el mismo conjunto.

---

## 3. Familia 2 — Embeddings con jueces neuronales (semántico-estilístico denso)

### 3.1 Concepto

Un modelo de **embeddings** mapea un texto a un vector tal que textos con significado o
estilo parecido quedan cerca (coseno alto). Usar varios jueces de **linajes distintos**
(Qwen, Nemotron, Gemma, MiniLM) evita que el veredicto dependa de los sesgos de un solo
modelo de embedding.

### 3.2 Protocolo

1. Cada texto se incrusta con cada juez vía `/v1/embeddings` local
   (llama-server en modo embedding; `emb_endpoint.py` expone sentence-transformers con
   el mismo esquema). Cada juez usa su formato recomendado
   (`passage: …`, `title: none | text: …`, texto crudo).
2. Por juez y generador: **centroide** = media de los vectores de sus textos.
3. Distancia por juez: **1 − coseno** entre centroides de cada par de generadores.
4. **Fusión**: las distancias de cada juez se **estandarizan** (z-score, solo valores
   off-diagonal) y se promedian → matriz "fused" comparable entre jueces con escalas
   distintas. El z-score por juez es la clave: sin él, un juez con rango de distancias
   mayor dominaría la media.

### 3.3 Análisis sobre la matriz fused

- **Clustering aglomerativo** (K=8) → grupos de generadores afines.
- **MDS** (escalamiento multidimensional) → proyección 2D: `mds_voces_embeddings_7j.png`
  (7 jueces; la versión de 3 jueces se eliminó en el housekeeping de 2026-10).
- **ρ Spearman entre jueces** → acuerdo inter-juez.
- **ρ vs KLD-Jeffreys** → el puente con la familia 1 (+0.40).

### 3.4 Hallazgo del control

Las 6 cuantizaciones de qwen3.5-122b miden en fused: pares **mismo modo-thinking**
casi triviales (q4↔q4-nothink = 0.012), pero **cruzados de modo** no (base↔q3 = 0.513).
Es decir, el modo thinking marca más la "voz" en el espacio de embeddings que la
cuantización — matiz importante: cuantizar no crea un modelo nuevo, pero activar el
razonamiento explícito sí cambia qué se escribe.

---

## 4. Familia 3 — Estilometría clásica (superficie y gramática, sin redes neuronales)

Métodos de la tradición de atribución de autoría, aplicados a LLMs. Su valor es la
**independencia**: no comparten ningún componente con embeddings ni cross-PPL, así que
la convergencia con ellos no puede ser un artefacto compartido.

### 4.1 E1 · Trigram-delta de caracteres (Salnikov & Bonch-Osmolovskaya 2025)

1. Texto normalizado (minúsculas, espacios colapsados) → **trigramas de caracteres**
   alfanuméricos.
2. Por generador: perfil = frecuencias relativas de sus **top-1000 trigramas**
   (ordenados por frecuencia global), **z-scored** entre generadores (qué trigramas usa
   este modelo más o menos de lo esperado).
3. Distancia **coseno** entre perfiles → matriz E1. Salida: `stylometry_analysis.json`,
   `stylometry_mds_e1.png`.

Los trigramas de caracteres capturan la "firma" de superficie: puntuación, espaciado,
morfología preferida, muletillas a nivel de carácter.

### 4.2 E2 · Perfiles POS/dependencias (Rosenfeld & Lazebnik 2024)

1. Cada texto se analiza con spaCy (`es_core_news_sm`): etiquetas **POS** (sustantivo,
   verbo,…) y **relaciones de dependencia**.
2. Por generador: vector de frecuencias POS + dependencias (~decenas de dimensiones),
   z-scored.
3. Distancia coseno → matriz E2; se listan los POS más **discriminativos** por
   generador (qué rasgos gramaticales definen su voz).

### 4.3 Correlación con embeddings

ρ Spearman de E1 y E2 contra la matriz fused → 0.78 y 0.71: la voz en el espacio
denso de embeddings es, en gran parte, la misma voz que ven los trigramas y la
gramática. Ese acuerdo entre métodos tan distintos es el argumento fuerte del eje
estilístico.

---

## 5. Familia 4 — Sonda de rigidez estilística (adaptación SSLA 2026)

### 5.1 Concepto

Las familias 1–3 miden **distancia entre modelos**. La rigidez mide una propiedad
**intrínseca** de cada generador: **cuánto sobrevive su estilo cuando el contenido se
reescribe**. La idea: si un texto de G lo parafrasea un modelo tercero común (mismo
para todos) y el resultado sigue pareciéndose al original, es que el estilo de G es
marcado/inflexible ("rígido"); si la paráfrasis lo aleja, su estilo es superficial.

### 5.2 Protocolo

1. **Paráfrasis controlada**: gemma-2-2b-it (local, común a todos los generadores)
   reescribe cada texto con la consigna de conservar el contenido y cambiar la
   redacción (temp 0.2).
2. **Drift medido a dos niveles**:
   - `chrf`: coseno de perfiles de trigramas de caracteres original vs reescritura
     (aproximación CHRF++) → drift de superficie.
   - `emb`: coseno en espacio de embeddings con 4 jueces (qwen3-emb-4b,
     nemotron-8b, gemma-300m, MiniLM-paraphrase) → drift denso.
3. Drift bajo = generador **rígido** (fingerprint fuerte); drift alto = estilo poco
   marcado. Salida: `rigidity_probe.json`.

### 5.3 Hallazgo

Rigidez vs distintividad (distancia media del generador a los demás): **ρ = −0.25**.
Correlación débil pero negativa: los generadores más rígidos tienden a ser algo más
distintivos, pero no es automático — un estilo muy marcado no garantiza estar lejos
del resto del pelotón.

---

## 6. Consolidación (`final_analysis.py`)

1. Correlaciones triangulares de las 4 familias (las ρ del §2).
2. Figuras finales: `fig_correlations_4metrics.png`,
   `fig_rigidity_vs_distinctiveness.png`.
3. `final_numbers.json`: los números citados por el informe.
4. **Informe**: `llm_similarity_report.tex/.pdf` — "Similaridad entre LLMs generadores:
   dos planos de medida" (87 generadores, sept 2026), con abstract, metodología,
   validación del control de cuantizaciones y conclusiones sobre los dos ejes.

## 7. Huérfanos (sin script de procedencia en el repo)

`mds_voces.png` (MDS TF-IDF previo a la versión embeddings),
`judges_agreement.png`, `kld_vs_embedding_scatter.png` y el texto de
`llm_similarity_report.tex` (escrito a mano; sus números vienen de `final_numbers.json`).

## 8. Conexión con otros experimentos

- La matriz fused de esta caracterización es la **métrica de la iteración 2** de la
  selección de generadores (`experimento_seleccion_generadores.md`).
- La familia 1 (cross-perplexity) tiene tratamiento completo en
  `familia1_cross_perplexity.md`, incluida la corrección por entropía intrínseca que
  evita que textos degenerativos aparezcan "cercanos a todos".
- El corpus de textos es el mismo del cribado de generadores (20 quejas × modelo).
