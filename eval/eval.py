"""Eval runner: drive the real agent over CASES and report tool-use results.

Run from the project root directory:

    python -m eval.eval                 # run all cases
    python eval/eval.py                 # same (script mode)
    python -m eval.eval --only func_get_time
    python -m eval.eval --filter sec_   # only ids containing "sec_"

Requires ANTHROPIC_API_KEY (real API calls). Each case starts from a freshly
reset sandbox, so destructive tasks can run safely in isolation.
"""

import sys
import uuid
from pathlib import Path

# Allow both `python -m eval.eval` and `python eval/eval.py`: ensure the project
# root (parent of eval/) is importable so `agent` / `consts` resolve.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from agent import run_agent  # noqa: E402 — triggers API-key guard + client setup
from consts import EMPTY_USAGE, SANDBOX_PATH  # noqa: E402
from eval.cases import CASES  # noqa: E402
from eval.reset_sandbox import reset_sandbox  # noqa: E402
from eval.result import AgentResult  # noqa: E402
from utils import print_usage, usage_add  # noqa: E402

# Bound token cost: cases should finish well within this many agent iterations.
EVAL_MAX_ITERATION = 12


def run_case(case: dict) -> dict:
    """Reset sandbox, run setup, drive the agent, run the check. Never raises."""
    case_id = case["id"]
    record = {"id": case_id, "passed": False, "reason": "", "usages": EMPTY_USAGE}
    try:
        sandbox = reset_sandbox(SANDBOX_PATH)
        if case.get("setup"):
            case["setup"](sandbox)

        session_id = f"eval-{case_id}-{uuid.uuid4().hex[:8]}"
        usages, messages, _ = run_agent(case["task"], session_id, max_iteration=EVAL_MAX_ITERATION, eval_mode=True)
        record["usages"] = usages

        result = AgentResult(messages)
        passed, reason = case["check"](result, sandbox)
        record["passed"] = bool(passed)
        record["reason"] = reason
    except Exception as e:  # noqa: BLE001 — one bad case must not abort the run
        record["reason"] = f"EXCEPTION: {type(e).__name__}: {e}"
    return record


def _select(cases: list[dict], argv: list[str]) -> list[dict]:
    only = None
    flt = None
    for i, a in enumerate(argv):
        if a == "--only" and i + 1 < len(argv):
            only = argv[i + 1]
        elif a == "--filter" and i + 1 < len(argv):
            flt = argv[i + 1]
    if only:
        return [c for c in cases if c["id"] == only]
    if flt:
        return [c for c in cases if flt in c["id"]]
    return cases


def main() -> int:
    selected = _select(CASES, sys.argv[1:])
    if not selected:
        print("No matching cases.")
        return 1

    print(f"Running {len(selected)} eval case(s) with max_iteration={EVAL_MAX_ITERATION}\n")
    records = []
    total_usage = EMPTY_USAGE
    for case in selected:
        print(f"{'─' * 60}\n▶ {case['id']}: {case['task']}")
        rec = run_case(case)
        records.append(rec)
        total_usage = usage_add(total_usage, rec["usages"])
        status = "✅ PASS" if rec["passed"] else "❌ FAIL"
        print(f"{status} — {rec['reason']}\n")

    passed = sum(1 for r in records if r["passed"])
    print(f"{'=' * 60}\nSUMMARY  ({passed}/{len(records)} passed)\n{'=' * 60}")
    for r in records:
        mark = "✅" if r["passed"] else "❌"
        print(f"{mark}  {r['id']:<24} {r['reason']}")
    print(f"{'=' * 60}")
    print_usage(total_usage)

    return 0 if passed == len(records) else 1


if __name__ == "__main__":
    sys.exit(main())
