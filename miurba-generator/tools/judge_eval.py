#!/usr/bin/env python3
"""Evalúa el juez del pipeline contra el gold del panel (gold_100).

Etiquetas derivadas de los 6 votos del panel (3 deepseek + 3 glm):
  - APTO     : 6/6 votos a favor            (accepted.jsonl)
  - NO APTO  : >=4/6 votos en contra         (rejected.jsonl)
  - dudoso   : 1-3 votos en contra           (se excluye del cálculo)

Cada vuelta evalúa el prompt ACTUAL del juez (prompts/validate_record.md) con el
modelo indicado, sobre los mismos casos, y añade una línea al marcador.

Uso:
  python tools/judge_eval.py --judge gemma4-12b-awq --tag v11-gemma4 \
      --base-url http://host:9292
"""
import argparse
import json
import os
import random
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.llm import LLMClient              # noqa: E402
from src.pipeline import validate_complaint  # noqa: E402

GOLD = Path(os.environ.get("MIURBA_EVAL_DIR", ROOT / "eval")) / "judge_bench/gold_100"
BIG = Path(os.environ.get("MIURBA_EVAL_DIR", ROOT / "eval")) / "judge_bench"


def load_gold_big(aptos_limit: int = 270, only_consensus: bool = True):
    """Banco grande (948 casos) con las anotaciones de deepseek y glm.

    Etiqueta = acuerdo de los dos anotadores: APTO si ambos lo aceptan, defecto
    si ambos lo rechazan. Los 61 casos en que discrepan se descartan. Se
    descartan tambien los escenarios imposibles ('ninguno' junto a otro riesgo).
    Devuelve (casos, excluidos) en el mismo formato que load_gold.
    """
    labels: dict[str, dict] = {}
    for path in sorted((BIG / "labels").glob("*_batch_*.jsonl")):
        family = "deepseek" if "deepseek" in path.name else "glm"
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            labels.setdefault(r["case_id"], {})[family] = r

    cases, aptos, defects, excluded = [], [], [], []
    for line in open(BIG / "cases.jsonl", encoding="utf-8"):
        r = json.loads(line)
        votes = labels.get(r["case_id"], {})
        if only_consensus and len(votes) < 2:
            continue
        verdicts = {v["apto"] for v in votes.values()}
        if len(verdicts) != 1:
            continue
        gold = verdicts.pop()
        scenario = r["scenario"]
        if has_impossible_risks(scenario):
            excluded.append(r["case_id"])
            continue
        row = (r["case_id"], r["model"], scenario, r["text"], gold)
        (aptos if gold else defects).append(row)

    random.Random(42).shuffle(aptos)
    return aptos[:aptos_limit] + defects, excluded


def _resolve_cases(args):
    """Carga el gold pedido: 'small' (panel de 100) o 'big' (948 anotados)."""
    if args.gold == "big":
        cases, excluded = load_gold_big(args.aptos_limit)
        load_gold.excluded = excluded
        return cases
    return load_gold()


def install_prompt_override(directory: str) -> None:
    """Sirve un validate_record.md alternativo sin tocar el pipeline.

    Copia la carpeta real de prompts a .tmp/prompt_variant/ y sobrescribe
    validate_record.md con la variante, para que todos los templates (examples,
    etc.) sigan resolviendose. Apunta prompt_loader.PROMPTS_DIR ahi una sola vez.
    """
    import shutil
    import src.prompt_loader as pl

    variant = Path(directory)
    if not variant.is_absolute():
        variant = ROOT / variant
    source = variant / "validate_record.md"
    if not source.exists():
        raise SystemExit(f"no hay validate_record.md en {variant}")

    staging = ROOT / ".tmp" / "prompt_variant"
    if staging.exists():
        shutil.rmtree(staging)
    shutil.copytree(pl.PROMPTS_DIR, staging)
    shutil.copyfile(source, staging / "validate_record.md")

    pl.PROMPTS_DIR = staging
    pl.load_template.cache_clear()
    print(f"prompt alternativo: {source} -> {staging}")


def load_gold():
    """Devuelve lista de (case_id, generador, scenario_plano, text, gold_bool).

    Descarta los casos cuyo escenario es imposible: 'ninguno' conviviendo con un
    riesgo concreto. Un texto no puede satisfacer los dos a la vez, asi que el
    juicio de 'risks' es indecidible y el caso no mide nada. El muestreo que los
    generaba esta corregido en src/sampling.py desde el commit 4bdb7ac.
    """
    raw = []
    for line in open(GOLD / "accepted.jsonl", encoding="utf-8"):
        r = json.loads(line)
        raw.append((r["case_id"], r["generador"], r["scenario"], r["text"], True))
    for line in open(GOLD / "rejected.jsonl", encoding="utf-8"):
        r = json.loads(line)
        against = sum(1 for v in r["votos"] if v["apto"] is False)
        if against >= 4:
            raw.append((r["case_id"], r["generador"], r["scenario"], r["text"], False))

    cases, excluded = [], []
    for case_id, gen, scenario, text, gold in raw:
        if has_impossible_risks(scenario):
            excluded.append(case_id)
            continue
        cases.append((case_id, gen, to_flat(scenario), text, gold))
    load_gold.excluded = excluded
    return cases


def has_impossible_risks(extractions: list[dict]) -> bool:
    """True si alguna extraccion pide 'ninguno' junto a un riesgo concreto."""
    for e in extractions:
        risks = e.get("risks") or []
        if "ninguno" in risks and len(risks) > 1:
            return True
    return False


def to_flat(extractions: list[dict]) -> list[dict]:
    """Convierte el escenario del corpus (formato extraction_schema) al formato
    plano que espera build_validation_prompt."""
    out = []
    for e in extractions:
        cat = e.get("category") or {}
        loc = e.get("location") or {}
        out.append({
            "category": cat.get("principal"),
            "subcategory": cat.get("subcategory"),
            "urgency": e.get("urgency"),
            "location_zone": loc.get("zone"),
            "location_detail": loc.get("detail") or "",
            "affected_elements": e.get("affected_elements") or [],
            "risks": e.get("risks") or ["ninguno"],
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", required=True)
    ap.add_argument("--base-url", required=True, help="OpenAI-compatible base URL")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=4000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--votes", type=int, default=1,
                    help="llamadas al juez por caso; decides por mayoria de los votos validos")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--stratified", type=int, default=0,
                    help="todos los defectos + N aptos, para cribar variantes barato")
    ap.add_argument("--gold", choices=("small", "big"), default="small",
                    help="small: panel de 130 casos; big: 948 casos anotados por deepseek+glm")
    ap.add_argument("--aptos-limit", type=int, default=270,
                    help="cuantos casos aptos del banco grande usar (los defectos van todos)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--prompt-dir", default=None,
                    help="carpeta con un validate_record.md alternativo (variantes de prompt)")
    args = ap.parse_args()

    if args.prompt_dir:
        install_prompt_override(args.prompt_dir)

    cases = _resolve_cases(args)
    if args.stratified:
        defects = [c for c in cases if not c[4]]
        aptos = [c for c in cases if c[4]][:args.stratified]
        cases = aptos + defects
    if args.limit:
        cases = cases[:args.limit]
    client = LLMClient(base_url=args.base_url, model=args.judge,
                       temperature=args.temperature, timeout=300, max_retries=3,
                       retry_delay=5.0, model_load_wait=180,
                       extra_params={"max_tokens": args.max_tokens})
    rows, lock = [], threading.Lock()

    def run(case):
        case_id, gen, scenario, text, gold = case
        votes, failures, issues, kinds, fields = [], [], [], [], {}
        for _ in range(max(1, args.votes)):
            try:
                valid, iss, degraded, raw = validate_complaint(client, text, scenario)
                if degraded or not (raw or {}).get("fields"):
                    reason = list(iss) or ["degraded sin issues"]
                    failures.append(reason)
                    if not issues:
                        issues = reason
                    continue
                votes.append(bool(valid))
                if not valid and not issues:
                    issues = list(iss)
                kinds.extend(d.get("kind") for d in (raw or {}).get("_diagnostics", []))
                fields = (raw or {}).get("fields") or fields
            except Exception as e:  # noqa: BLE001
                reason = [f"crash:{type(e).__name__}: {str(e)[:120]}"]
                failures.append(reason)
                if not issues:
                    issues = reason
        if votes:
            verdict = sum(votes) * 2 > len(votes)   # mayoria estricta
        else:
            verdict = False                          # sin votos validos: fail-closed
        with lock:
            rows.append({"case_id": case_id, "generador": gen, "gold": gold,
                         "judge": verdict, "issues": list(issues)[:6],
                         "kinds": [k for k in kinds if k] + ["fail"] * len(failures),
                         "votos": votes, "fallos": len(failures),
                         "motivos": failures,
                         "campos_graves": [k for k, v in fields.items()
                                           if str(v).lower() == "grave"]})
            print(f"  {case_id} gold={gold} judge={verdict} votos={votes} "
                  f"fallos={len(failures)} issues={list(issues)[:2]}", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(run, cases))

    tp = sum(1 for r in rows if not r["gold"] and not r["judge"])
    fn = sum(1 for r in rows if not r["gold"] and r["judge"])
    tn = sum(1 for r in rows if r["gold"] and r["judge"])
    fp = sum(1 for r in rows if r["gold"] and not r["judge"])
    n = len(rows)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0

    by_gen = {}
    for gen in sorted({r["generador"] for r in rows}):
        sub = [r for r in rows if r["generador"] == gen]
        by_gen[gen] = {"n": len(sub),
                       "aciertos": sum(1 for r in sub if r["gold"] == r["judge"]),
                       "colados": sum(1 for r in sub if not r["gold"] and r["judge"]),
                       "rechazados": sum(1 for r in sub if r["gold"] and not r["judge"])}

    summary = {"tag": args.tag, "judge": args.judge, "temperature": args.temperature,
               "votes": args.votes, "prompt_dir": args.prompt_dir, "gold": args.gold,
               "n": n, "n_positivos": tn + fp, "n_negativos": tp + fn,
               "excluidos_ninguno_mal_usado": getattr(load_gold, "excluded", []),
               "accuracy": round((tp + tn) / n, 4),
               "precision_defecto": round(prec, 4), "recall_defecto": round(rec, 4),
               "f1_defecto": round(2 * prec * rec / (prec + rec), 4) if prec + rec else 0.0,
               "colados": fn, "falsos_rechazos": fp,
               "llamadas_sin_respuesta": sum(r["fallos"] for r in rows),
               "motivos_fallo": dict(Counter(m for r in rows for f in r.get("motivos", []) for m in f)),
               "casos_sin_ningun_voto": sum(1 for r in rows if not r["votos"]),
               "casos_divididos": sum(1 for r in rows if r["votos"] and 0 < sum(r["votos"]) < len(r["votos"])),
               "fallos_juez": dict(Counter(k for r in rows for k in r["kinds"] if k)),
               "por_generador": by_gen,
               "desacuerdos": [r for r in rows if r["gold"] != r["judge"]]}

    out = GOLD / "results" / f"{args.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n== {args.tag} · juez {args.judge} (T={args.temperature}, {args.votes} voto(s)) "
          f"sobre {n} casos ({tn + fp} aptos / {tp + fn} defectuosos del panel)")
    if getattr(load_gold, "excluded", []):
        print(f"   excluidos por 'ninguno' mal usado: {len(load_gold.excluded)} "
              f"({', '.join(load_gold.excluded)})")
    print(f"   exactitud {summary['accuracy']:.1%} | defecto P {prec:.3f} R {rec:.3f} "
          f"F1 {summary['f1_defecto']:.3f}")
    print(f"   defectos colados {fn} | aptos rechazados {fp}")
    print(f"   llamadas sin respuesta {summary['llamadas_sin_respuesta']} | "
          f"casos sin ningun voto {summary['casos_sin_ningun_voto']} | "
          f"casos divididos {summary['casos_divididos']}")
    if summary["motivos_fallo"]:
        print(f"   motivos: {summary['motivos_fallo']}")
    print(f"   fallos del juez: {summary['fallos_juez']}")
    for g, s in by_gen.items():
        print(f"   {g:24s} n={s['n']:3d} aciertos {s['aciertos']:3d} "
              f"colados {s['colados']:2d} rechazados {s['rechazados']:2d}")
    print(f"\n   -> {out}")


if __name__ == "__main__":
    main()
