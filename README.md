# miUrba

Generación sintética y extracción estructurada de quejas vecinales.
Dos herramientas independientes sobre contratos compartidos:

- **`miurba-generator/`** — genera datasets sintéticos de quejas: texto del
  vecino (LLM) y ground truth estructurado.
- **`miurba-extractor/`** — extrae estructura de quejas reales o sintéticas
  (texto) con un LLM, conforme a un JSON Schema.

La raíz del repo solo contiene **contratos**: los JSON Schema de datos y el
vocabulario de dominio (`schemas/taxonomy.yaml`). Cada subproyecto posee su
propio `config/` con su config por defecto y ejemplos.

## Estructura

```
mi-urba/
├── schemas/                     # contratos compartidos
│   ├── complaint_schema.json
│   ├── extraction_schema.json
│   └── taxonomy.yaml            # vocabulario de dominio (taxonomías, personas, pesos)
├── miurba-generator/
│   ├── cli.py                   # generate, merge
│   ├── config/                  # config.yaml + ejemplos (smoke, corpus, merge)
│   └── docker-compose.yaml
└── miurba-extractor/
    ├── cli.py                   # extract
    ├── config/                  # config.yaml + ejemplo json_schema
    └── docker-compose.yaml
```

## Flujo de datos

```
miurba-generator                          miurba-extractor
  escenario ─► LLM ─► user_text                user_text ─► LLM
       ▼                                          ▼
  complaints.jsonl + ground_truth.jsonl + metadata.jsonl    extractions.jsonl
       └────────────── entrada del extractor ──────────────┘
```

El nexo entre ambas es el formato de datos (`schemas/`), no código.

### Generator — `cli.py generate`

```mermaid
flowchart TB

    subgraph CFG["Carga de config y contratos"]
        direction TB
        START(["python cli.py generate"]) --> LOAD["config/config.yaml<br/>endpoints · models · generation · paths"]
        LOAD --> DOTENV[".env del subproyecto<br/>secrets como ${VAR}"]
        LOAD --> TAXCHECK{"claves de taxonomía<br/>inline en el config"}
        TAXCHECK -->|"sí"| FASTFAIL["FAIL-FAST — el dominio<br/>solo vive en taxonomy.yaml"]
        TAXCHECK -->|"no"| TAX["schemas/taxonomy.yaml<br/>categorías · elementos · zonas · riesgos<br/>urgencias · expresiones · personas · pesos"]
        TAX --> MERGETAX["merge de taxonomía en el config"]
        MERGETAX --> VALTAX["validadores fail-fast<br/>taxonomy · coherencia · expresiones"]
        VALTAX -->|"inconsistente"| FASTFAIL
    end

    subgraph PLAN["Plan de lotes y clientes LLM"]
        direction TB
        VALTAX --> GENPLAN["plan — models+num_records<br/>o batches por modelo"]
        GENPLAN --> CHECKVAL{"validate activo"}
        CHECKVAL -->|"sin validator"| FASTFAIL
        CHECKVAL -->|"true"| JUDGE["LLMClient juez — distinto<br/>del generador"]
        GENPLAN --> GENCLI["LLMClient generador<br/>retries · model_load_wait · extra_params"]
    end

    subgraph REC["Por registro"]
        direction TB
        GENCLI --> SCEN["muestreo de escenario<br/>categoría por pesos · urgencia · zona<br/>riesgos · elementos coherentes<br/>persona · modifier · multi-problema"]
        SCEN --> PROMPT["build_generation_prompt<br/>plantillas + expresiones de urgencia"]
        PROMPT --> CHAT["LLM — user_text + timestamp en JSON"]
        CHAT --> EJSON["extract_json"]
        EJSON -->|"vacío · inválido"| REJGEN["rejected.jsonl"]
        EJSON --> HYG["higiene SIEMPRE — leaks de ids<br/>snake_case · markdown · 12-grama"]
        HYG -->|"leak"| REGEN
        HYG --> VALACT{"validate activo"}
        VALACT -->|"no"| WRITE
        VALACT -->|"sí"| VCALL["juez — veredicto por campos<br/>fail-closed"]
        VCALL -->|"ok"| WRITE["escritura con lock<br/>complaints · ground_truth · metadata"]
        VCALL -->|"rechazado"| REGEN{"presupuesto de reintentos"}
        REGEN -->|"queda"| TURN["regeneración multi-turn<br/>con judge_comment"]
        TURN --> CHAT
        REGEN -->|"agotado"| DISC["descartado — rejected.jsonl<br/>+ conversations.jsonl"]
    end

    subgraph FIN["Fusión"]
        direction TB
        WRITE --> MERGEOUT["merge por modelo → data/raw/<br/>ordenado por id"]
    end

    classDef failnode fill:#fce8e6,stroke:#d93025,color:#5f1b16
    classDef llmnode fill:#fef7e0,stroke:#e37400,color:#5c3d00
    class FASTFAIL,REJGEN,DISC failnode
    class CHAT,VCALL,JUDGE,GENCLI llmnode
```

- La higiene determinista corre **siempre**; el juez semántico es opcional y
  **fail-closed**: un juez inalcanzable nunca aprueba.
- El ground truth se construye en código desde el escenario muestreado
  (alineado con `schemas/extraction_schema.json`); nunca lo escribe el LLM.

### Extractor — `cli.py extract`

```mermaid
flowchart TB

    subgraph CFGX["Config y schema"]
        direction TB
        STARTX(["python cli.py extract"]) --> LOADX["config/config.yaml<br/>+ taxonomy de schemas/"]
        LOADX --> SCHEMA["extraction_schema.json con enums<br/>sincronizados de la taxonomía"]
        LOADX --> MODELP["models → endpoint + extra_params"]
    end

    subgraph MODE["Salida estructurada (por modelo)"]
        direction TB
        MODELP --> SMODE{"structured_output"}
        SMODE -->|json_object| JOBJ["response_format json_object"]
        SMODE -->|json_schema| JSCHEMA["response_format json_schema<br/>el servidor fuerza el schema"]
        SMODE -->|"off"| FREEMODE["texto libre"]
    end

    subgraph LOOPX["Por queja"]
        direction TB
        FREEMODE --> ONEX
        JOBJ --> ONEX
        JSCHEMA --> ONEX
        ONEX["build_extraction_prompt<br/>enums de taxonomía en el prompt"] --> CHATX["LLM — extractions en JSON"]
        CHATX -->|"error · vacío · JSON inválido"| FAILED["failed — no se escribe"]
        CHATX --> NORM["normaliza complaint_id e hispanismos"]
        NORM --> VALIDX["jsonschema SIEMPRE<br/>contra extraction_schema.json"]
        VALIDX -->|"no conforme"| FAILED
        VALIDX -->|"conforme"| WRITEX["extractions.jsonl"]
    end

    classDef failnode fill:#fce8e6,stroke:#d93025,color:#5f1b16
    classDef llmnode fill:#fef7e0,stroke:#e37400,color:#5c3d00
    class FAILED failnode
    class CHATX llmnode
```

- El modo `structured_output` restringe el muestreo **en el servidor**; la
  validación local contra el schema (enums sincronizados desde
  `taxonomy.yaml`) se aplica **siempre**.

### Pipeline end-to-end

```mermaid
sequenceDiagram
    autonumber
    actor U as Usuario
    participant GC as cli.py generate
    participant GL as LLM generador
    participant JV as LLM juez
    participant FS as JSONL en data/
    participant EC as cli.py extract
    participant XL as LLM extractor

    U->>GC: generate
    activate GC
    GC->>GC: config + taxonomy + .env — validadores fail-fast
    loop por registro del plan
        GC->>GC: muestreo de escenario · persona · modifier
        GC->>GL: prompt de generación
        GL-->>GC: JSON con user_text y timestamp
        GC->>GC: higiene determinista
        opt validate activo
            GC->>JV: prompt de validación
            JV-->>GC: veredicto con evidencia
            alt rechazado
                GC->>GL: regeneración multi-turn
                GL-->>GC: nuevo user_text
            end
        end
        alt aceptado
            GC->>FS: complaints · ground_truth · metadata
        else presupuesto agotado
            GC->>FS: rejected.jsonl + conversations.jsonl
        end
    end
    GC->>FS: merge por modelo → data/raw
    deactivate GC

    U->>EC: extract
    activate EC
    EC->>EC: schema con enums sincronizados
    loop por queja pendiente
        EC->>XL: prompt de extracción
        XL-->>EC: JSON de extractions
        EC->>EC: normaliza + jsonschema SIEMPRE
        alt válido
            EC->>FS: annotated/extractions.jsonl
        else inválido
            EC->>EC: cuenta como failed
        end
    end
    deactivate EC
```

## Requisitos

- Python 3.10+ (probado con 3.12)
- Un endpoint LLM OpenAI-compatible (vLLM, llama.cpp/llama-swap, Ollama…)

## Inicio rápido

```bash
# 1. Configura endpoints y modelos en <subproyecto>/config/config.yaml.
#    Los secretos van en <subproyecto>/.env (copia .env.example) y se
#    referencian como ${VAR}.

# 2. Generar quejas sintéticas
cd miurba-generator
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python cli.py generate                # usa config/config.yaml

# 3. Extraer estructura de las quejas generadas: apunta raw_dir del
#    config del extractor a la salida del generador y ejecuta
#    python cli.py extract    # usa config/config.yaml
```

Salidas por defecto: `miurba-generator/data/<modelo>/raw/` en el generador y
`miurba-extractor/data/annotated/extractions.jsonl` en el extractor.

## Docker

Cada subproyecto tiene su propio compose:

```bash
cd miurba-generator && docker compose run --rm miurba-generator generate --config /app/config/config.yaml
cd miurba-extractor  && docker compose run --rm miurba-extractor  extract  --config /app/config/config.yaml
```

Montan `./config` (configs), `../schemas` (contratos), `./data` y `./logs`.

## Configuración

Cada subproyecto lee `config/config.yaml` (default) o lo que se pase con
`--config`. El vocabulario de dominio (taxonomías, personas, pesos) vive en
`schemas/taxonomy.yaml` y se referencia con `taxonomy:`; declararlo inline en
un config es un error fail-fast. Detalle en los README de cada subproyecto.

Ejemplos: `miurba-generator/config/{config-smoke,config-corpus,config-merge}.yaml`
y `miurba-extractor/config/config-jsonschema.yaml`.

## Testing

Ejecuta `pytest` **dentro** de cada subproyecto (los paquetes `src` son
independientes y colisionan si se lanzan desde la raíz):

```bash
cd miurba-generator && python -m pytest tests/
cd miurba-extractor && python -m pytest tests/
```

No requieren red.
