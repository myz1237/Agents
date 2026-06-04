"""Reset the sandbox to a known fixture state — internal eval helper.

Wipes the sandbox directory and recreates a deterministic set of files so every
eval / manual test run starts from an identical baseline. The fixture contains:

- a ``mini-project/`` package (normal source + a long file for range reads),
- hidden files (``.env``, ``.hidden_file.txt``, ``mini-project/.gitignore``),
- ordinary top-level files (text, binary, a replace-test fixture),
- an empty directory.

Usage (run from the ``hello_world/`` directory so ``consts`` is importable):

    python -m eval.reset_sandbox

or programmatically:

    from eval.reset_sandbox import reset_sandbox
    reset_sandbox()
"""

import shutil
import sys
from pathlib import Path

# Allow running both as a module (`python -m eval.reset_sandbox`) and as a plain
# script (`python eval/reset_sandbox.py`). When run as a script, sys.path[0] is
# this file's dir (eval/), not the project root, so `consts` wouldn't resolve —
# add the project root (the parent of eval/) to the path before importing it.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from consts import SANDBOX_PATH  # noqa: E402  (import after sys.path bootstrap)


def _build_test_replace() -> str:
    """Generate the replace-test fixture: 8 functions, each printing "Hello".

    Kept programmatic so a reset always restores the original (all-"Hello")
    state, no matter what previous string_replace runs did to the file.
    """
    blocks: list[str] = []
    for i in range(4):
        suffix = "" if i == 0 else f"_{i}"
        blocks.append(f'def hi{suffix}():\n    print("before hi")\n    print("Hello")')
        blocks.append(f'def world{suffix}():\n    print("World")\n    print("Hello")')
    return "\n\n\n".join(blocks) + "\n"


_README = """# Mini Project

A tiny sample project used as a sandbox fixture for evals.

## Layout
- `main.py` — entry point, prints a couple of string transforms.
- `utils.py` — the string helpers.
- `test_utils.py` — unit tests for the helpers.
- `report_builder.py` — a deliberately long module for range-read tests.
"""

_UTILS = """def reverse_string(s):
    return s[::-1]


def count_words(text):
    return len(text.split())


def capitalize_words(text):
    return " ".join(word.capitalize() for word in text.split())
"""

_MAIN = """from utils import reverse_string, count_words

if __name__ == "__main__":
    text = "Hello World from Agent"
    print(reverse_string(text))
    print(count_words(text))
"""

_TEST_UTILS = '''import unittest

from utils import capitalize_words, count_words, reverse_string


class TestUtils(unittest.TestCase):
    def test_reverse_string_basic(self):
        self.assertEqual(reverse_string("hello"), "olleh")

    def test_reverse_string_empty(self):
        self.assertEqual(reverse_string(""), "")

    def test_count_words(self):
        self.assertEqual(count_words("a b c"), 3)

    def test_capitalize_words(self):
        self.assertEqual(capitalize_words("hello world"), "Hello World")


if __name__ == "__main__":
    unittest.main()
'''

_REPORT_BUILDER = '''"""A deliberately long, self-contained module for range-read testing.

The functionality here is intentionally unimportant — it exists so you can ask
an LLM to read a specific slice and observe how it picks start_line/end_line.
"""

from collections import Counter


def build_sales_report(records: list[dict]) -> str:
    """Build a plain-text sales report from a list of order records."""
    lines: list[str] = ["SALES REPORT", "=" * 40]
    total = 0.0
    by_region: Counter = Counter()
    for rec in records:
        order_total = sum(item["qty"] * item["price"] for item in rec["items"])
        total += order_total
        by_region[rec["region"]] += order_total
        lines.append(f"{rec['id']:<10} {rec['customer']:<20} {order_total:>10.2f}")
    lines.append("-" * 40)
    lines.append(f"{'TOTAL':<31} {total:>10.2f}")
    lines.append("")
    lines.append("BY REGION")
    for region, amount in by_region.most_common():
        lines.append(f"  {region:<10} {amount:>10.2f}")
    return "\\n".join(lines)


def summarize_status(records: list[dict]) -> dict:
    """Count how many orders are in each status bucket."""
    counts: Counter = Counter(rec.get("status", "unknown") for rec in records)
    return dict(counts)


def top_customers(records: list[dict], n: int = 3) -> list[tuple]:
    """Return the top-n customers by total spend."""
    spend: Counter = Counter()
    for rec in records:
        order_total = sum(item["qty"] * item["price"] for item in rec["items"])
        spend[rec["customer"]] += order_total
    return spend.most_common(n)


if __name__ == "__main__":
    sample = [
        {
            "id": "ORD-1001",
            "customer": "Acme Corp",
            "region": "EMEA",
            "items": [{"sku": "A1", "qty": 3, "price": 9.99}],
            "status": "shipped",
        },
        {
            "id": "ORD-1002",
            "customer": "Globex",
            "region": "APAC",
            "items": [{"sku": "B2", "qty": 1, "price": 49.0}],
            "status": "pending",
        },
    ]
    print(build_sales_report(sample))
    print(summarize_status(sample))
    print(top_customers(sample))
'''

# relative POSIX path -> text content
TEXT_FILES: dict[str, str] = {
    # --- ordinary top-level files ---
    "file.txt": "This is a normal text file.\n",
    "ok.txt": "ok\n",
    "new_file_2.txt": "Second sample file.\nUsed for list/read tests.\n",
    "test_replace.py": _build_test_replace(),
    # --- hidden top-level files ---
    ".hidden_file.txt": "hidden content here\n",
    ".env": "API_KEY=do-not-read-me\nDEBUG=true\n",
    # --- mini-project ---
    "mini-project/README.md": _README,
    "mini-project/main.py": _MAIN,
    "mini-project/utils.py": _UTILS,
    "mini-project/test_utils.py": _TEST_UTILS,
    "mini-project/report_builder.py": _REPORT_BUILDER,
    # hidden file inside the project
    "mini-project/.gitignore": "__pycache__/\n*.pyc\n",
}

# relative POSIX path -> binary content
BINARY_FILES: dict[str, bytes] = {
    "bin.dat": bytes([0x00, 0x01, 0x02, 0x03]),
}

# directories that should exist even when empty
EMPTY_DIRS: list[str] = ["empty_folder"]


def reset_sandbox(root: Path = SANDBOX_PATH) -> Path:
    """Wipe ``root`` and recreate the fixture. Returns the sandbox path.

    Guarded so it only ever deletes a directory literally named ``sandbox`` —
    a misconfigured ``SANDBOX_PATH`` should never wipe an unrelated directory.
    """
    root = root.resolve()
    if root.name != "sandbox":
        raise ValueError(f"refusing to reset: {root} is not a 'sandbox' directory")

    # Wipe existing contents so we start clean; if the dir doesn't exist yet
    # (e.g. fresh checkout), create it — together with any missing parents.
    if root.exists():
        shutil.rmtree(root)
        print(f"Removed existing sandbox: {root}")
    else:
        print(f"Sandbox did not exist, creating: {root}")
    root.mkdir(parents=True, exist_ok=True)

    for rel, text in TEXT_FILES.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")

    for rel, blob in BINARY_FILES.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(blob)

    for rel in EMPTY_DIRS:
        (root / rel).mkdir(parents=True, exist_ok=True)

    return root


if __name__ == "__main__":
    path = reset_sandbox()
    file_count = len(TEXT_FILES) + len(BINARY_FILES)
    print(f"Sandbox reset at: {path}")
    print(f"  {file_count} files, {len(EMPTY_DIRS)} empty dir(s)")
