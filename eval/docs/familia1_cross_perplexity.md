# Familia 1 · Cross-Perplexity entre generadores — Conceptos, glosario y metodología

> Fuente: `eval/experiments/seleccion_modelos/cross_perplexity_unified.py`, `eval/experiments/seleccion_modelos/analyze_cross_perplexity.py`,
> `eval/experiments/seleccion_modelos/run_cross_perplexity.sh`, `eval/experiments/seleccion_modelos/run_gateway_matrix.sh`, `eval/llama_cpp_prompt_logprobs.patch`.
> Resultados: `eval/results/diversity/cross_perplexity*.json`, `xppl_gen_cache*.json`.

---

## 1. La idea en una frase

> **Si el texto que produce el modelo G le resulta "predecible" (poco sorprendente) al modelo E,
> es que G y E escriben en la misma distribución.** La cross-perplexity mide esa sorpresa.

Cada modelo se usa dos veces: como **generador** (escribe quejas vecinales) y como
**evaluador** (puntúa cuánto le sorprende el texto ajeno). El resultado es una matriz
`H[E][G]` — una "matriz de confusiones estilísticas" entre todos los pares de modelos.

---

## 2. Glosario

### 2.1 Token y tokenización
Un LLM no procesa caracteres ni palabras: procesa **tokens**, fragmentos de texto de su
vocabulario (BPE/SentencePiece). `"fontanería"` puede ser 1 token para un modelo y 3 para otro.
**Implicación clave:** las probabilidades por token de dos modelos distintos **no son
directamente comparables** palabra a palabra; por eso se normaliza por token y se derivan
métricas robustas (ver §4).

### 2.2 Probabilidad y log-probabilidad (logprob)
Un modelo lingüístico define, para cada contexto, una distribución de probabilidad sobre el
siguiente token: `P(token | contexto)`. En la práctica se trabaja con
`log P` (**logprob**, número ≤ 0) porque:
- evita underflow al multiplicar cientos de probabilidades (se convierten en sumas),
- es lo que el servidor expone directamente.

### 2.3 NLL — Negative Log-Likelihood (log-verosimilitud negativa)
Dado un texto de `L` tokens `t1..tL`:

```
NLL = −(1/L) · Σᵢ log P(tᵢ | t₁..tᵢ₋₁, prompt)
```

Es la **media de logprobs con signo cambiado** → un número ≥ 0 en **nats/token**
(logaritmo natural). Interpretación:
- **NLL baja** → el evaluador "esperaba" esos tokens → el texto le resulta natural.
- **NLL alta** → el evaluador se sorprende → el texto le resulta ajeno/extraño.

### 2.4 Entropía (de Shannon) — el concepto base
La **entropía** H(P) de una distribución P mide su incertidumbre media:

```
H(P) = − Σₓ P(x) · log P(x)
```

Es el número esperado de nats necesarios para codificar un símbolo extraído de P
usando un código **óptimo diseñado para la propia P**. Baja = distribución
predecible/concentrada; alta = distribución dispersa. Para un LLM, la entropía de su
distribución de siguiente-token mide "cuánta duda tiene" en cada paso.

### 2.5 Entropía cruzada — y en qué se distingue de la entropía
La **entropía cruzada** de una distribución Q (el "modelo") respecto a otra P (la real)
es:

```
H(P, Q) = − Σₓ P(x) · log Q(x)
```

Es el coste esperado de codificar muestras de P usando un código **óptimo para Q en
vez de para P**. Dos diferencias cruciales con la entropía de §2.4:

1. **Intervienen dos distribuciones** (P y Q), no una. La entropía H(P) es el caso
   especial H(P, P).
2. **Siempre H(P, Q) ≥ H(P)**: usar un código diseñado para Q nunca puede costar menos
   que usar el óptimo para P. El exceso es exactamente la divergencia KL:

```
H(P, Q) = H(P) + KL(P ‖ Q)      ← desigualdad de Gibbs
```

**En nuestro experimento**, no conocemos P ni Q en forma cerrada (son distribuciones
sobre secuencias completas de texto). Lo que hacemos es **estimar H(P_G, Q_E) por
muestreo**: generamos muestras reales de P_G (los textos que G escribe) y evaluamos su
log-verosimilitud media bajo Q_E (el modelo E). Esa estimación por muestras es
exactamente la NLL de §2.3:

```
H[P_G escrito, puntúado por E] ≈ (1/L) Σᵢ −log P_E(tᵢ | …) = NLL
```

Por eso NLL ≡ entropía cruzada estimada, y por eso KLD (§4.2) se deriva como
H[P_G, Q_E] − H[P_G, Q_G]: restando la diagonal eliminamos el término H(P_G), que
es la entropía propia del generador y no nos interesa, quedándonos solo con KL(P_G‖Q_E).

### 2.6 Perplejidad (PPL) y cross-perplejidad
La **perplejidad** es la exponencial de la entropía cruzada: `PPL = exp(H)`. Cambia la
unidad, no el contenido: mientras H está en nats/token, la PPL se lee como "el modelo se
comporta como si en cada paso eligiera entre N tokens igual de probables". PPL=1 →
predicción perfecta (la diagonal ideal E==G con temp=0); PPL=1000 → el token real
estaba entre ~1000 candidatos igual de verosímiles. Ojo: la PPL es **exponencial**, así
que diferencias moderadas en H son diferencias enormes en PPL — para comparar y
correlacionar entre modelos, el pipeline trabaja siempre en H (nats), nunca en PPL.

La **cross-perplejidad** es esa misma cantidad medida *cruzada*: la muestra la sacó P_G
(el texto lo escribió G) pero el código usado es Q_E (lo puntúa E). Es decir,
`cross-PPL(G→E) = exp(H(P_G, Q_E))`. De ahí el nombre del experimento (y de las
variantes `-xppl` de los servidores del gateway: llama.cpp compilado con el parche de
`prompt_logprobs`).

### Resumen de equivalencias

| Cantidad | Definición | Relación |
|---|---|---|
| Entropía H(P) | −Σ P log P | caso especial H(P,P) |
| Entropía cruzada H(P,Q) | −Σ P log Q | H(P,Q) = H(P) + KL(P‖Q) |
| NLL (media por token) | −(1/L) Σ log P_E(tᵢ‖…) | **estimación por muestreo de H(P_G, Q_E)** |
| Perplejidad | exp(H) | misma info, escala exponencial |
| Cross-perplejidad | exp(H(P_G, Q_E)) | la PPL de la entropía cruzada estimada |
| KL(P‖Q) | Σ P log(P/Q) | el exceso de la entropía cruzada sobre la entropía |

⚠️ En los JSON guardados (`mean_logprob`) el valor almacenado es el logprob medio
(≤ 0); el análisis lo convierte a H = −mean_logprob (≥ 0, nats/token). Ver §4.

### 2.7 Condicionamiento por el prompt propio
Cada texto se genera con **el prompt de producción real de G** (el del generador de quejas,
iter 6). Al puntuar, E recibe `prompt_de_G + texto_de_G` y solo se mide el texto. Esto
garantiza que la diferencia H[E][G] − H[G][G] aísle el **estilo del contenido generado**,
no diferencias de plantilla.

### 2.8 nats vs bits
Con logaritmo natural, la unidad es el **nat** (1 nat = 1/ln2 ≈ 1.44 bits). El repo usa
nats en todo el pipeline.

---

## 3. Protocolo del experimento: cómo se mide cada celda

Para construir la matriz hay que repetir, para cada par (evaluador E, generador G) y cada
escenario, el mismo procedimiento básico:

1. **Cargar G** (uno a la vez; llama-swap gestiona la VRAM) y generar una queja vecinal
   con su prompt de producción, en un escenario presampleado idéntico para todos los modelos.
2. **Puntuar ese texto con E**: pedir a E los logprobs del texto completo condicionado al
   prompt, y quedarse con la media sobre los tokens del contenido.
3. Guardar la celda `H[E][G]` y repetir con el siguiente modelo.

Ese procedimiento simple esconde varios problemas metodológicos que, si no se controlan,
distorsionan la matriz. Estos son los problemas y las decisiones que se tomaron para cada uno:

### 3.1 ¿Qué parte del texto se puntúa? — Solo el contenido, nunca el prompt

**Problema.** La forma ingenua es calcular la NLL media de todo lo que se le envía a E:
prompt + respuesta. Pero el prompt es idéntico para todos los modelos, mientras que cada
E lo lee con distinta soltura (unos modelos han visto más textos de ese formato que otros).
Ese efecto de "familiaridad con la plantilla" y el propio peso de la longitud del prompt se
mezclan con la señal que buscamos (cómo reacciona E al *texto de G*), y pueden dominarla.

**Decisión.** Registrar dónde empieza el contenido generado y promediar los logprobs
solo sobre esos tokens. El prompt sirve únicamente como contexto de condicionamiento
(§2.7).

### 3.2 ¿Cómo genera G sus textos? — Greedy con muestreo neutro

**Problema.** La interpretación de la matriz depende de que la generación sea
reproducible y "limpia":

- Con temperatura > 0 el mismo modelo produce textos distintos en cada ejecución y las
  celdas dejan de ser comparables entre runs.
- llama.cpp aplica por defecto un `repeat_penalty=1.1` (y presencia/frecuencia > 0 en
  algunas configs). Eso hace que el token efectivamente generado **no** sea el que el
  modelo consideraba más probable. Consecuencia sutil pero fatal: cuando un modelo
  puntúa su propio texto, la probabilidad que obtiene es menor que la del token argmax
  (la penalización ya desvió la elección), y la diagonal de la matriz — que debería ser
  el punto de referencia PPL ≈ 1 — sale inflada de forma inconsistente entre modelos.

**Decisión.** Generar con `temperature=0` (greedy: siempre el token más probable) y con
las penalizaciones desactivadas (`repeat_penalty=1.0`, `presence=0`, `frequency=0`), de
modo que el token muestreado coincide con el argmax del modelo y la diagonal recupera su
significado.

### 3.3 ¿Cómo se puntuá la diagonal? — Reutilizando los tokens generados

**Problema.** La celda E==G es la ancla de toda la matriz (es la que define las KLD,
ver §4.2), así que no puede contener ruido. Si al puntuarla se vuelve a pasar el texto
por el servidor, una re-tokenización ligeramente distinta (o un redondeo del servidor)
cambia el valor y rompe la simetría esperada.

**Decisión.** Cuando G genera, se guardan en un **gen-cache** sus propios tokens junto
con su logprob medio ya calculado. Cuando toca la celda E==G, se reutilizan esos valores
directamente en lugar de recalcularlos.

### 3.4 ¿Cómo se obtienen los logprobs de E? — `prompt_logprobs` (1 petición) vs incremental (O(L))

**Problema.** La API estándar de completions de llama.cpp no devuelve la probabilidad de
texto *ya escrito*: devuelve la distribución del *siguiente* token. La forma ingenua de
puntuar un texto de L tokens es alimentar los primeros i tokens y leer la probabilidad
del token i+1: **L peticiones por texto**. Con L≈400 tokens y decenas de miles de celdas,
es inviable (minutos por celda).

**Decisión.** Se parcheó llama.cpp (`eval/llama_cpp_prompt_logprobs.patch`, imagen en
`eval/llama_cpp_xppl.Dockerfile`; los modelos `-xppl` del gateway usan ese build) para
que `prompt_logprobs` devuelva, en **una sola petición**, el logprob de cada token del
prompt. El script intenta este método y, si el servidor no está parcheado, cae
automáticamente al incremental (`--method auto`), y cada celda registra qué método usó.

### 3.5 ¿Qué se guarda de cada medición? — Granularidad por escenario, no solo medias

**Problema.** Guardar solo la media final por celda (E, G) impide auditar el resultado:
una media puede esconder un escenario atípico, un texto degenerado o un fallo puntual.

**Decisión.** Cada celda almacena la medición **por escenario** (`mean_logprob`,
`n_tokens`) además de los agregados, junto con `n_misses` (generaciones o puntuaciones
fallidas) y `method`. Eso permite análisis pareado (comparar E1 vs E2 sobre el mismo
estímulo) y detectar outliers antes de promediar.

**Campos por celda:** `mean_ppl`, `mean_logprob`, `n_scenarios`, `n_tokens`, `n_misses`, `method`.

---

## 4. Métricas derivadas (`analyze_cross_perplexity.py`)

La NLL cruda mezcla **tres efectos** que hay que separar:
(a) la distancia verdadera entre distribuciones, (b) la confianza/calibración del
evaluador (su diagonal), y (c) la entropía intrínseca de cada generador (un texto
degenerativo y repetitivo es trivialmente predecible **para todos**).

### 4.1 H[A][B] — entropía cruzada en nats/token
```
H[A][B] = −mean_logprob[A][B]
```
El punto de partida; todo lo demás se deriva de aquí.

### 4.2 KLD[A][B] — divergencia KL por muestreo
```
KLD[A][B] = H[A][B] − H[B][B]   ≈  KL(P_B ‖ P_A)
```
La línea base es la **entropía del propio generador sobre su texto** (H[B][B]), no la
autoconfianza del evaluador. Es la estimación por muestras de cuántos nats extra
cuesta codificar la salida de B usando el modelo A. Efecto clave: un generador con
textos degenerativos (entropía intrínseca baja) ya no aparece "artificialmente
cercano" a todos.

### 4.3 J — divergencia de Jeffreys (KL simetrizada)
```
J[A][B] = KLD[A][B] + KLD[B][A]
```
Simétrica y ≥ 0 en esperanza. Es la que se usa para las correlaciones con las demás
familias de métricas (ρ KLD↔Emb = +0.40 verificado).

### 4.4 dH y su versión simetrizada
```
dH[A][B]     = H[A][B] − H[A][A]        (desde el punto de vista del evaluador)
S[A][B]      = (dH[A][B] + dH[B][A]) / 2
```
Alternativa que normaliza por la **autoconfianza del evaluador** en vez de por la del
generador. Nota del código: los valores negativos son posibles y con significado —
los textos de cada generador tienen entropías intrínsecas distintas.

### 4.5 Distinct-2 (diversidad léxica, flag de degeneración)
```
ratio = |bigramas únicos| / |bigramas totales|   (media por generador)
```
Sobre texto en minúsculas tokenizado por `\w+`, ignorando textos de <4 palabras.
Un ratio bajo delata degeneración/repetición, que es exactamente el caso que KLD
corrige y que conviene detectar de forma explícita.

### 4.6 Lectura de la matriz (criterio del script)
- **Fila (evaluador fijo) alta** → la salida de B es lejana a la de todos los demás.
- **Fila baja** → la salida de B "podría haber salido de cualquiera" → **redundante**:
  no aporta diversidad al conjunto de generadores.

---

## 5. Variantes de ejecución

| Run | Alcance | Fichero de resultados |
|-----|---------|----------------------|
| Exploración local | 4 modelos pequeños locales (gemma-2-2b-it, falcon3-1b, qwen3-0.6b, llama-3.2-1b) | `cross_perplexity.json`, `cross_perplexity_v4.json` — **eliminados en housekeeping (2026-10)**: fase exploratoria sin dependientes; los scripts activos usan el cache `_gateway_full` |
| Gateway inicial | 4 modelos `-xppl` del gateway | `cross_perplexity_gateway.json` — **eliminado en housekeeping (2026-10)**: contenido en la matriz `_full` |
| Gateway "big" | 3 modelos grandes 70B-class | `cross_perplexity_gateway_big.json` |
| Gateway "full" | 26 modelos `-xppl` (lista construida desde los registrados en llama-swap, excluyendo arquitecturas eliminadas de llama.cpp moderno como mixtral) | `cross_perplexity_gateway_full.json` + `xppl_gen_cache_gateway_full.json` |

Parámetros de entorno: `GATEWAY` (default `http://192.168.10.11:9292`), `N_SCENARIOS`
(default 3), `REQ_TIMEOUT` 900 s. Ejecución estrictamente secuencial, un modelo cargado
a la vez vía llama-swap.

**Requisito de infraestructura:** el servidor de llama.cpp debe estar compilado con el
parche `prompt_logprobs` (`eval/llama_cpp_prompt_logprobs.patch`,
`eval/llama_cpp_xppl.Dockerfile`); si no, el método incremental O(L) como fallback.

---

## 6. Validación integrada (comprobaciones de sanidad)

- **Diagonal ≈ 1 de PPL** cuando E==G con tokens del gen-cache: valida que penalizaciones
  neutras + temp=0 están activos.
- **Celdas missing / `n_misses`** registradas por celda: fallos de generación o puntuación
  quedan trazados, no ocultos en la media.
- **Análisis pareado por escenario**: al guardar cada (E, G, escenario) se pueden
  comparar pares con el mismo estímulo y detectar outliers por texto.

---

## 7. Conexión con las otras familias

| Familia | Mide | Relación verificada con KLD |
|---------|------|------------------------------|
| F2 Embeddings (7 jueces) | semántico-estilístico denso | **ρ = +0.40** → eje semántico compartido |
| F3a Trigram-delta | estilo de superficie | ρ = +0.38 |
| F3b POS/dep | estilo gramatical | ρ = +0.25 |
| F4 Rigidez | resistencia del estilo a paráfrasis | vía correlación con distintividad (−0.25) |

Conclusión del conjunto: KLD captura sobre todo el **eje semántico** ("qué se dice"),
ortogonal al **eje estilístico** ("cómo se escribe") donde convergen trigramas, POS y
embeddings (ρ 0.71–0.83 entre sí).
