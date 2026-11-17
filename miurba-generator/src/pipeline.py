"""
Pipeline functions — synthetic complaint GENERATION.

Orchestrates scenario sampling, LLM calls (text) and file I/O to produce a
dataset of raw complaints:

  - complaints.jsonl   (id, user_text, timestamp)
  - ground_truth.jsonl  (complaint_id, extractions)
  - metadata.jsonl      (complaint_id, persona, modifier, community)
  - rejected.jsonl      (discarded records with stage and reason)
  - conversations.jsonl (per-record generation transcript)

The LLM produces ONLY user_text + timestamp.
"""

import json
import logging
import os
import random
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import config as cfg
from . import contracts
from .llm import LLMClient
from .prompt_loader import render as _render_prompt
from .prompts import build_generation_prompt, build_validation_prompt
from .sampling import (
    sample_community,
    sample_persona,
    sample_modifier,
    sample_scenario,
    should_be_multi_problem,
)

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    """UTC timestamp for provenance records."""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_jsonl_ids(path: Path, key: str = "id") -> list[str]:
    """Read the ids of a JSONL file, tolerating truncated or corrupt lines.

    A run killed mid-write can leave a partial line; a corrupt line must not
    break the resume pass, it is skipped with a warning.
    """
    ids: list[str] = []
    if not path.exists():
        return ids
    with open(path, encoding="utf-8") as f:
        for number, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                logger.warning(f"{path.name}:{number} unreadable line, skipped")
                continue
            ids.append(str(record.get(key, "")))
    return ids


def _write_jsonl_atomic(path: Path, records: list[dict]) -> None:
    """Write a JSONL file through a temp file + atomic replace."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _filter_jsonl(path: Path, keep: set[str], key: str) -> int:
    """Rewrite a JSONL file keeping only the given ids; return how many were dropped."""
    if not path.exists():
        return 0
    kept: list[dict] = []
    parsed = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            parsed += 1
            if str(record.get(key, "")) in keep:
                kept.append(record)
    dropped = parsed - len(kept)
    if dropped:
        _write_jsonl_atomic(path, kept)
    return dropped


def reconcile_outputs(raw_path: Path, ground_truth_path: Path,
                      metadata_path: Path) -> set[str]:
    """Keep only the records written to all three files; repair the rest.

    `complaints`/`ground_truth`/`metadata` are three separate appends, so a
    process killed between them leaves a record without ground truth. Those
    ids are removed from every file so a later resume regenerates them.
    """
    complaint_ids = _read_jsonl_ids(raw_path, "id")
    gt_ids = _read_jsonl_ids(ground_truth_path, "complaint_id")
    meta_ids = _read_jsonl_ids(metadata_path, "complaint_id")
    complete = set(complaint_ids) & set(gt_ids) & set(meta_ids)
    dropped = {
        "complaints": _filter_jsonl(raw_path, complete, "id"),
        "ground_truth": _filter_jsonl(ground_truth_path, complete, "complaint_id"),
        "metadata": _filter_jsonl(metadata_path, complete, "complaint_id"),
    }
    dropped = {k: v for k, v in dropped.items() if v}
    if dropped:
        logger.warning(f"Dropped incomplete records to keep the corpus consistent: {dropped}")
    return complete


def normalize_timestamp(value: str) -> str | None:
    """Return the timestamp as canonical ISO 8601, or None if unparseable.

    Models emit offsets as `+0000`, `+00:00` or `Z`, so the value is normalised
    before parsing (this also keeps the code valid on Python 3.10, where
    `fromisoformat` accepts neither `Z` nor `+0000`). A naive value is assumed
    to be UTC; a record whose timestamp cannot be parsed is rejected, because
    the corpus contract requires a real date-time.
    """
    from datetime import datetime, timezone
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    text = re.sub(r"([+-]\d{2})(\d{2})$", r"\1:\2", text)
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.isoformat()


# ── Validation (2-stage) ──────────────────────────────────────────────

# ── Hygiene gate (deterministic, always on) ────────────────────

_HYGIENE_PATTERNS = [
    (re.compile(r"\bQ\d{3,6}(_\d+)?(\.(png|jpe?g|webp))?\b", re.I),
     "internal id/filename in text"),
    (re.compile(r"\bcategor[ií]a\b|\bsubcategor[ií]a\b", re.I),
     "taxonomy words in text"),
    (re.compile(r"\*\*|``|<think>|\{\"|user_text"), "prompt/markdown artifacts"),
    (re.compile(r"\b[a-záéíóúñ]+_[a-záéíóúñ]+\b"), "snake_case token"),
]


def hygiene_failures(user_text: str) -> list[str]:
    """Deterministic integrity checks applied to every generated text.

    Returns a list of human-readable failures; empty list = clean text.
    Unlike the LLM judge, this runs ALWAYS (independent of `validate`).

    Scope: leaks and artefacts only (internal ids/filenames, taxonomy labels,
    prompt echoes, markdown, snake_case tokens, degeneration loops). Style is
    NOT checked here: all-caps or short messages are legitimate expressions of
    the sampled persona/modifier, so rejecting them would contradict the
    stylistic diversity the corpus is built for.
    """
    fails = []
    for pat, msg in _HYGIENE_PATTERNS:
        if pat.search(user_text):
            fails.append(msg)
    words = user_text.split()
    if len(words) >= 24:
        grams = [" ".join(words[i:i + 12]) for i in range(len(words) - 11)]
        if len(grams) != len(set(grams)):
            fails.append("repeated 12-gram (degeneration loop)")
    return fails


def _judge_failure_kind(exc: Exception, response) -> str:
    """Classify why a judge call produced no usable verdict (for diagnosis)."""
    if isinstance(exc, json.JSONDecodeError) or isinstance(exc, ValueError):
        return "unparseable"
    if isinstance(exc, (KeyError, IndexError, TypeError)):
        return "malformed_shape"
    name = type(exc).__name__.lower()
    if "timeout" in name or "connection" in name:
        return "transport"
    if "http" in name:
        return "http_error"
    return name


def _response_facts(response) -> dict:
    """Finish reason and token usage of a chat completion, for failure analysis."""
    facts = {}
    try:
        choice = (response or {}).get("choices", [{}])[0]
        facts["finish_reason"] = choice.get("finish_reason")
    except Exception:
        pass
    try:
        usage = (response or {}).get("usage") or {}
        facts["completion_tokens"] = usage.get("completion_tokens")
        facts["prompt_tokens"] = usage.get("prompt_tokens")
    except Exception:
        pass
    return facts


def _judge_raw_snippet(client, response, limit: int = 4000) -> str:
    """Best-effort raw content of a judge reply, for the diagnostics record."""
    if response is None:
        return ""
    try:
        return (client.extract_content(response) or "")[:limit]
    except Exception:
        try:
            return json.dumps(response)[:limit]
        except Exception:
            return ""


def _call_judge(client, messages, diagnostics: list, max_attempts: int = 2):
    """Call the judge, retrying once on an unusable verdict, recording every failure.

    Raises the last exception when no attempt produced a parseable dict, so the
    caller can mark the record as unvalidated (fail-closed).
    """
    last_exc = None
    for attempt in range(1, max_attempts + 1):
        response = None
        try:
            response = client.chat(messages)
            data = client.extract_json(response)
            if not isinstance(data, dict):
                raise ValueError(f"judge verdict is {type(data).__name__}, expected object")
            diagnostics.append({"attempt": attempt, "kind": "ok"})
            return data
        except Exception as e:  # noqa: BLE001 - classified and re-raised below
            last_exc = e
            diagnostics.append({
                "attempt": attempt,
                "kind": _judge_failure_kind(e, response),
                "error": str(e)[:200],
                "raw": _judge_raw_snippet(client, response),
                **_response_facts(response),
            })
            logger.warning(
                f"Judge attempt {attempt}/{max_attempts} failed "
                f"({diagnostics[-1]['kind']}): {str(e)[:120]}"
            )
    raise last_exc


def validate_complaint(client: LLMClient, user_text: str, problems: list[dict]) -> tuple[bool, list[str], bool]:
    """Validate that user_text faithfully describes the scenario.

    Fail-closed: a validation error (LLM call, malformed JSON) returns
    is_valid=False plus a degraded flag, so a broken judge can never let
    unvalidated records into the corpus.

    Returns (is_valid, issues_list, degraded, raw_judge_response).
    """
    messages = build_validation_prompt(user_text, problems)
    diagnostics: list[dict] = []
    try:
        data = _call_judge(client, messages, diagnostics)
        # Strict: only a literal true counts. A judge returning "valid":
        # "false"/"no"/"0" (strings) must NOT pass (bool("false") is True).
        # A judge that omits the key is not approving either.
        is_valid = data.get("valid") is True
        issues = data.get("issues") or []
        if not isinstance(issues, list):
            issues = [str(issues)]
        # Per-field verdicts are the primary signal when present: any 'grave' or a
        # problem-count mismatch makes the record invalid regardless of `valid`.
        fields = data.get("fields")
        if isinstance(fields, dict):
            severe = [f for f, v in fields.items() if str(v).lower() == "grave"]
            issues = list(dict.fromkeys(issues + severe))
            if severe:
                is_valid = False
        invented = data.get("inventado") or {}
        if isinstance(invented, dict) and any(invented.get(k) for k in invented):
            is_valid = False
            issues = list(dict.fromkeys(issues + [f"inventado:{k}" for k in invented if invented.get(k)]))
        n_txt = data.get("n_problems_texto")
        if isinstance(n_txt, int) and n_txt != len(problems):
            is_valid = False
            issues = list(dict.fromkeys(issues + [f"n_problems:{n_txt}!={len(problems)}"]))
        # Mechanical urgency check: the level the judge actually found in the text
        # must match the scenario level, problem by problem when the count agrees.
        detected = data.get("urgencias_detectadas") or data.get("urgencia_detectada")
        expected_seq = [str(p.get("urgency")) for p in problems if p.get("urgency")]
        if isinstance(detected, list) and len(detected) == len(expected_seq):
            for pos, (det, exp) in enumerate(zip(detected, expected_seq), 1):
                if str(det).strip().lower() != exp.lower():
                    is_valid = False
                    issues = list(dict.fromkeys(issues + [f"urgency[{pos}]:{det}!={exp}"]))
        elif isinstance(detected, str) and expected_seq and detected.strip().lower() not in expected_seq:
            is_valid = False
            issues = list(dict.fromkeys(issues + [f"urgency:{detected}!={sorted(set(expected_seq))}"]))
        # Feed the judge's own evidence back into the retry prompt: a field name is
        # a weak hint, a quoted fragment tells the model what to fix.
        evidence = data.get("evidencia")
        quoted = data.get("expresion_citada")
        if not is_valid:
            if isinstance(evidence, str) and evidence.strip():
                issues = list(dict.fromkeys(issues + [f"evidencia: {evidence.strip()[:160]}"]))
            if isinstance(quoted, str) and quoted.strip():
                issues = list(dict.fromkeys(issues + [f"expresion_citada: {quoted.strip()[:80]}"]))
        if diagnostics:
            data["_diagnostics"] = diagnostics
        return is_valid, issues, False, data
    except Exception as e:
        logger.warning(f"Validation call failed: {e}, assuming INVALID (fail-closed)")
        return False, [f"judge_error:{_judge_failure_kind(e, None)}"], True, {
            "error": str(e)[:200],
            "error_kind": _judge_failure_kind(e, None),
            "_diagnostics": diagnostics,
        }


# ── Generation ──────────────────────────────────────────────────

def generate_complaint(
    client: LLMClient,
    record_id: str,
    scenario: dict | None = None,
    validate: bool = False,
    max_validation_retries: int = 3,
    validator_client: LLMClient | None = None,
    validation_log: list | None = None,
    reject_sink=None,
    conversation_sink=None,
    provenance: dict | None = None,
) -> tuple[dict, dict, dict] | None:
    """
    Generate a single raw complaint record.

    Returns:
        Tuple of (complaint_dict, ground_truth, metadata) or None on failure.
    """
    if scenario is None:
        scenario = sample_scenario()

    # Provenance defaults: the CLI always passes the real values; direct callers
    # (tests, scripts) get a complete but empty record so the contract still holds.
    provenance = {"model": "", "model_id": "", "seed": None,
                  "config_hash": "", "prompt_fingerprints": {},
                  **(provenance or {})}
    if validate and validator_client is None:
        raise ValueError(
            "validate=True requires an explicit judge client "
            "(generation.validator); the generator is never its own judge"
        )

    community = sample_community()
    persona = sample_persona()
    modifier = sample_modifier()

    # Multi-problem?
    if should_be_multi_problem():
        n_problems = random.choices([2, 3], weights=[0.75, 0.25], k=1)[0]
    else:
        n_problems = 1

    problems = [scenario] + [sample_scenario() for _ in range(n_problems - 1)]

    # Build prompt and call LLM (text only: user_text + timestamp)
    messages = build_generation_prompt(
        record_id=record_id,
        problems=problems,
        community=community,
        persona=persona,
        modifier=modifier,
    )

    try:
        response = client.chat(messages)
        content = client.extract_content(response)

        if not content:
            logger.warning(f"[{record_id}] Empty content, skipping")
            if reject_sink is not None:
                reject_sink({"id": record_id, "stage": "generation",
                             "reason": "empty_content", "detail": "no content in reply",
                             "attempts": 1, "scenario": problems,
                             "last_text": "", "timestamp": "",
                             **_response_facts(response)})
            return None

        try:
            data = client.extract_json(response)
        except json.JSONDecodeError as e:
            logger.warning(f"[{record_id}] JSON parse failed: {e}")
            if reject_sink is not None:
                reject_sink({"id": record_id, "stage": "generation",
                             "reason": "invalid_json", "detail": str(e)[:200],
                             "attempts": 1, "scenario": problems,
                             "last_text": content[:4000], "timestamp": "",
                             **_response_facts(response)})
            return None

    except RuntimeError as e:
        logger.error(f"[{record_id}] LLM call failed: {e}")
        if reject_sink is not None:
            reject_sink({"id": record_id, "stage": "generation",
                         "reason": "llm_error", "detail": str(e)[:200],
                         "attempts": 1, "scenario": problems,
                         "last_text": "", "timestamp": ""})
        return None

    user_text = data.get("user_text", "")
    timestamp = normalize_timestamp(data.get("timestamp", "")) or ""

    if not user_text:
        logger.warning(f"[{record_id}] No user_text in LLM output")
        if reject_sink is not None:
            reject_sink({"id": record_id, "stage": "generation",
                         "reason": "no_user_text", "detail": "JSON without user_text",
                         "attempts": 1, "scenario": problems,
                         "last_text": str(data)[:2000], "timestamp": ""})
        return None

    if not timestamp:
        logger.warning(f"[{record_id}] Missing or unparseable timestamp")
        if reject_sink is not None:
            reject_sink({"id": record_id, "stage": "contract",
                         "reason": "schema_violation",
                         "detail": "timestamp: missing or unparseable date-time",
                         "attempts": 1, "scenario": problems,
                         "last_text": user_text, "timestamp": ""})
        return None

    # Single acceptance gate. The FIRST generation and EVERY regeneration are
    # validated identically, in the same order: (1) deterministic leak/artefact
    # check, (2) semantic judge. A candidate that fails either one is
    # regenerated from the same scenario; the record is written only if some
    # candidate passes both. If the budget is exhausted the record is
    # DISCARDED (never written) and the resume pass retries it later with a
    # fresh scenario. A judge that cannot be reached never counts as approval.
    judge = validator_client
    if validate and judge is None:
        raise ValueError(
            "validate=True requires an explicit judge client "
            "(generation.validator); the generator is never its own judge"
        )
    max_attempts = max_validation_retries if validate else max_validation_retries + 2
    accepted = False
    history: list[tuple[str, str]] = []
    transcript: list[dict] = []

    for attempt in range(max_attempts + 1):
        failure = None
        detail = ""

        judge_raw = None
        leaks = hygiene_failures(user_text)
        if leaks:
            failure = "leak"
            detail = "; ".join(leaks)
        elif validate:
            is_valid, issues, degraded, judge_raw = validate_complaint(judge, user_text, problems)
            if validation_log is not None:
                validation_log.append(
                    "degraded" if degraded else ("ok" if is_valid else "rejected")
                )
            if degraded:
                failure = "judge_unavailable"
                detail = "judge call failed; not treated as approval"
            elif not is_valid:
                failure = "rejected"
                detail = ", ".join(map(str, issues)) or "unknown"

        transcript.append({
            "attempt": attempt + 1,
            "user_text": user_text,
            "timestamp": timestamp,
            "hygiene_failures": leaks,
            "judge": judge_raw if validate else None,
            "issues": detail,
            "failure": failure,
        })

        if failure is None:
            accepted = True
            break

        logger.warning(
            f"[{record_id}] {failure} ({detail}); "
            f"regenerating (attempt {attempt + 1}/{max_attempts})"
        )
        if attempt == max_attempts:
            break
        # Feedback-driven retry as a real conversation: the rejected generation and
        # the judge's comment become assistant/user turns, so the model sees its own
        # previous answer and what was wrong with it.
        issues_for_repair = [i.strip() for i in re.split(r"[;,]", detail or "") if i.strip()]
        history.append((
            json.dumps({"user_text": user_text, "timestamp": timestamp}, ensure_ascii=False),
            _render_prompt(
                "judge_comment",
                issues="; ".join(issues_for_repair) if issues_for_repair else "no especificados",
                previous_hint="",
                n_problems=len(problems),
            ),
        ))
        retry_messages = build_generation_prompt(
            record_id=record_id,
            problems=problems,
            community=community,
            persona=persona,
            modifier=modifier,
            history=history,
        )
        try:
            response = client.chat(retry_messages)
            data = client.extract_json(response)
            regen_text = data.get("user_text", "")
            if not regen_text:
                logger.warning(f"[{record_id}] regeneration produced no user_text")
                break
            user_text = regen_text
            timestamp = normalize_timestamp(data.get("timestamp", "")) or timestamp
        except (RuntimeError, json.JSONDecodeError) as e:
            logger.error(f"[{record_id}] regeneration LLM call failed: {e}")
            break

    if not accepted:
        logger.error(
            f"[{record_id}] discarded: no candidate passed validation "
            f"after {max_attempts} regeneration attempts"
        )
        if conversation_sink is not None:
            conversation_sink({"id": record_id, "accepted": False, "scenario": problems,
                               "attempts": len(transcript), "turns": transcript,
                               "persona": (persona or {}).get("id"),
                               "modifier": (modifier or {}).get("id"),
                               })
        if reject_sink is not None:
            reject_sink({
                "id": record_id,
                "stage": "validation",
                "reason": failure or "unknown",
                "detail": detail,
                "attempts": max_attempts + 1,
                "scenario": problems,
                "last_text": user_text,
                "timestamp": timestamp,
            })
        return None

    complaint = {
        "id": record_id,
        "user_text": user_text,
        "timestamp": timestamp,
    }

    # Ground truth: aligned with extraction_schema.json
    extractions = [
        {
            "category": {"principal": p["category"], "subcategory": p["subcategory"]},
            "affected_elements": p["affected_elements"],
            "location": {"zone": p["location_zone"], "detail": p.get("location_detail", "")},
            "risks": p["risks"],
            "urgency": p["urgency"],
        }
        for p in problems
    ]

    ground_truth = {
        "complaint_id": record_id,
        "extractions": extractions,
    }

    metadata = {
        "complaint_id": record_id,
        "persona": persona["id"] if persona else None,
        "modifier": modifier["id"] if modifier else None,
        "community": community,
        "provenance": {**provenance, "generated_at": _now_iso()},
    }

    # Output contract: a record that does not conform to the published schemas
    # is rejected (and audited) instead of entering the corpus.
    violations = contracts.validate_record(complaint, ground_truth, metadata)
    if violations:
        logger.error(f"[{record_id}] contract violations: {'; '.join(violations)}")
        if reject_sink is not None:
            reject_sink({
                "id": record_id,
                "stage": "contract",
                "reason": "schema_violation",
                "detail": "; ".join(violations)[:2000],
                "attempts": len(transcript),
                "scenario": problems,
                "last_text": user_text,
                "timestamp": timestamp,
            })
        return None

    if conversation_sink is not None:
        conversation_sink({"id": record_id, "accepted": True, "scenario": problems,
                           "attempts": len(transcript), "turns": transcript,
                           "persona": (persona or {}).get("id"),
                           "modifier": (modifier or {}).get("id"),
                           })

    return complaint, ground_truth, metadata


def generate_dataset(
    client: LLMClient,
    num_records: int,
    start_id: int = 1,
    output_dir: str = "data",
    raw_subdir: str = "raw",
    complaints_file: str = "complaints.jsonl",
    ground_truth_file: str = "ground_truth.jsonl",
    metadata_file: str = "metadata.jsonl",
    resume: bool = True,
    validate: bool = False,
    max_validation_retries: int = 3,
    validator_client: LLMClient | None = None,
    num_workers: int = 1,
    provenance: dict | None = None,
) -> dict:
    """
    Generate raw complaint dataset.

    Produces:
      - <output_dir>/<raw_subdir>/<complaints_file>
      - <output_dir>/<raw_subdir>/<ground_truth_file>
      - <output_dir>/<raw_subdir>/<metadata_file>
    """
    raw_path = Path(output_dir) / raw_subdir / complaints_file
    ground_truth_path = Path(output_dir) / raw_subdir / ground_truth_file
    metadata_path = Path(output_dir) / raw_subdir / metadata_file
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    # resume=False means start fresh: truncate previous outputs
    if not resume:
        for f in (raw_path, ground_truth_path, metadata_path):
            if f.exists():
                f.write_text("")
                logger.info(f"Start fresh: truncated {f}")

    # Resume: only ids complete in all three files count as done. A record
    # written to complaints but not to ground_truth/metadata (process killed
    # between appends) is dropped from the files and regenerated, so the corpus
    # never carries a half-written record.
    existing_ids: set[str] = set()
    if resume:
        existing_ids = reconcile_outputs(raw_path, ground_truth_path, metadata_path)
        if existing_ids:
            logger.info(f"Resuming: {len(existing_ids)} complete records already exist")

    generated = 0
    failed = 0
    last_logged = [0]
    validation_log: list = []
    write_lock = threading.Lock()
    reject_path = Path(output_dir) / raw_subdir / "rejected.jsonl"
    conversation_path = Path(output_dir) / raw_subdir / "conversations.jsonl"
    if not resume:
        for p_ in (reject_path, conversation_path):
            if p_.exists():
                p_.write_text("")

    def _reject_sink(rec: dict) -> None:
        """Persist every discarded record: the corpus must report its own yield."""
        with write_lock:
            with open(reject_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def _conversation_sink(rec: dict) -> None:
        """Persist the full generator/judge transcript of every record."""
        with write_lock:
            with open(conversation_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def _generate_and_write(record_id: str) -> bool:
        """Generate one record and append it to the JSONL files (thread-safe).

        Any unexpected failure is contained to the record: a malformed model
        reply must never abort a campaign that already produced records.
        """
        try:
            return _generate_and_write_unsafe(record_id)
        except Exception as e:  # noqa: BLE001 - per-record isolation
            logger.exception(f"[{record_id}] unexpected failure: {e}")
            _reject_sink({"id": record_id, "stage": "pipeline",
                          "reason": "unexpected_error", "detail": str(e)[:500],
                          "attempts": 0, "scenario": [], "last_text": "",
                          "timestamp": ""})
            return False

    def _generate_and_write_unsafe(record_id: str) -> bool:
        """Generate one record and append it to the JSONL files (thread-safe)."""
        result = generate_complaint(
            client, record_id,
            validate=validate,
            max_validation_retries=max_validation_retries,
            validator_client=validator_client,
            validation_log=validation_log,
            reject_sink=_reject_sink,
            conversation_sink=_conversation_sink,
            provenance=provenance,
        )
        if result is None:
            logger.warning(f"[{record_id}] Generation failed")
            return False

        complaint, ground_truth, metadata = result
        with write_lock:
            with open(raw_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(complaint, ensure_ascii=False) + "\n")
            with open(ground_truth_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(ground_truth, ensure_ascii=False) + "\n")
            with open(metadata_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(metadata, ensure_ascii=False) + "\n")
        return True

    pending = [f"Q{i:04d}" for i in range(start_id, start_id + num_records)
               if f"Q{i:04d}" not in existing_ids]
    generated += num_records - len(pending)  # resumed records count as done

    if num_workers > 1:
        logger.info(f"Generating with {num_workers} parallel workers...")
        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            futures = {pool.submit(_generate_and_write, rid): rid for rid in pending}
            for n, future in enumerate(as_completed(futures), 1):
                if future.result():
                    generated += 1
                else:
                    failed += 1
                if n % 10 == 0:
                    logger.info(
                        f"Progress: {n}/{len(pending)} processed, {failed} failed, "
                        f"tokens: in={client.stats['tokens_in']:,}, "
                        f"out={client.stats['tokens_out']:,}"
                    )
    else:
        for record_id in pending:
            logger.info(f"[{record_id}] Generating ({generated + 1}/{num_records})...")
            if _generate_and_write(record_id):
                generated += 1
            else:
                failed += 1

            if generated % 10 == 0 and generated != last_logged[0]:
                last_logged[0] = generated
                logger.info(
                    f"Progress: {generated}/{num_records} generated, "
                    f"{failed} failed, "
                    f"tokens: in={client.stats['tokens_in']:,}, "
                    f"out={client.stats['tokens_out']:,}"
                )

    from collections import Counter
    validation_counts = Counter(validation_log)

    return {
        "generated": generated,
        "failed": failed,
        "total_tokens_in": client.stats["tokens_in"],
        "total_tokens_out": client.stats["tokens_out"],
        "raw_path": str(raw_path),
        "ground_truth_path": str(ground_truth_path),
        "metadata_path": str(metadata_path),
        "validation": dict(validation_counts) if validate else None,
    }


# ── Merge (multi-model batches) ─────────────────────────────────

def merge_outputs(
    output_dir: str = "data",
    batch_dirs: list[str] | None = None,
    raw_subdir: str = "raw",
    complaints_file: str = "complaints.jsonl",
    ground_truth_file: str = "ground_truth.jsonl",
    metadata_file: str = "metadata.jsonl",
) -> dict:
    """
    Merge per-model output directories into final <output_dir>/<raw_subdir>/.

    Records are deduplicated by id (first occurrence wins) so a re-plan that
    reassigns id ranges cannot silently duplicate the corpus, and the result is
    written atomically: a failed merge never truncates a previous good merge.
    """
    raw_out = Path(output_dir) / raw_subdir
    model_dirs = [Path(output_dir) / d / raw_subdir for d in (batch_dirs or [])]
    present = [d for d in model_dirs if d.is_dir()]
    if not present:
        raise FileNotFoundError(
            f"merge: no per-model output directory found under {output_dir} "
            f"(expected {', '.join(d.name for d in model_dirs) or 'none'})"
        )
    missing = [d.name for d in model_dirs if d not in present]
    if missing:
        logger.warning(f"merge: model directories not found, skipped: {', '.join(missing)}")

    merged: dict[str, list[dict]] = {
        complaints_file: [], ground_truth_file: [], metadata_file: []}
    seen: dict[str, set[str]] = {name: set() for name in merged}
    key_of = {complaints_file: "id", ground_truth_file: "complaint_id",
              metadata_file: "complaint_id"}

    for model_path in present:
        for fname in merged:
            fpath = model_path / fname
            if not fpath.exists():
                continue
            key = key_of[fname]
            for line in fpath.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning(f"merge: unreadable line in {fpath}, skipped")
                    continue
                record_id = str(record.get(key, ""))
                if record_id in seen[fname]:
                    logger.warning(f"merge: duplicate {key} {record_id} in {fpath}, skipped")
                    continue
                seen[fname].add(record_id)
                merged[fname].append(record)

    for fname in merged:
        merged[fname].sort(key=_id_num)

    complaints_path = raw_out / complaints_file
    ground_truth_path = raw_out / ground_truth_file
    metadata_path = raw_out / metadata_file

    raw_out.mkdir(parents=True, exist_ok=True)
    _write_jsonl_atomic(complaints_path, merged[complaints_file])
    _write_jsonl_atomic(ground_truth_path, merged[ground_truth_file])
    _write_jsonl_atomic(metadata_path, merged[metadata_file])

    counts = {name: len(records) for name, records in merged.items()}
    if len(set(counts.values())) > 1:
        logger.error(f"merge: the three files disagree in length: {counts}")
    else:
        logger.info(f"Merged {counts[complaints_file]} records → {complaints_path}")

    return {
        "complaints": counts[complaints_file],
        "ground_truth": counts[ground_truth_file],
        "metadata": counts[metadata_file],
        "complaints_path": str(complaints_path),
        "ground_truth_path": str(ground_truth_path),
        "metadata_path": str(metadata_path),
    }


def _id_num(rec: dict) -> int:
    try:
        return int(str(rec.get("id", rec.get("complaint_id", "Q0"))).lstrip("Q"))
    except (ValueError, AttributeError):
        return 0