"""
Stress Test Suite per TaxiAgent v4 — 15 scenari automatizzati.

Esegui con:
    python test_agent_stress.py

Richiede Ollama in esecuzione (llama3.2:3b).
Non richiede il bot Telegram.
"""

import sys
import json
import time
from dataclasses import dataclass, field
from typing import Optional, List

sys.path.insert(0, r'C:\Users\andre\Desktop\Progetto_Accenture')

from llm_tool.agent import get_agent

# ─── Test Infrastructure ──────────────────────────────────────────────────────

@dataclass
class TestCase:
    name: str
    message: str
    expected_intent: Optional[str] = None       # 'predict' | 'trend' | 'oos'
    expected_zone_id: Optional[int] = None       # location_id in final params
    expected_hour: Optional[int] = None          # hour in final params
    expected_dow: Optional[int] = None           # day_of_week in final params
    max_response_len: int = 600                  # chars
    has_candidates: bool = False                 # disambiguates?
    prev_params: dict = field(default_factory=dict)  # simulated session state
    history: list = field(default_factory=list)

RESET = "\033[0m"
GREEN = "\033[92m"
RED   = "\033[91m"
CYAN  = "\033[96m"
YELLOW = "\033[93m"


def check(label: str, condition: bool, detail: str = "") -> bool:
    icon = f"{GREEN}✓{RESET}" if condition else f"{RED}✗{RESET}"
    suffix = f"  ({detail})" if detail else ""
    print(f"    {icon} {label}{suffix}")
    return condition


def run_test(tc: TestCase, agent) -> bool:
    print(f"\n{CYAN}[{tc.name}]{RESET} {tc.message!r}")
    t0 = time.time()
    result = agent.chat(tc.message, chat_history=tc.history, current_params=tc.prev_params)
    elapsed = time.time() - t0

    text       = result.get("text", "")
    candidates = result.get("candidates", [])
    params     = result.get("params", {})

    print(f"    ⏱  {elapsed:.1f}s  |  len={len(text)}")
    print(f"    Response preview: {text[:120].replace(chr(10), ' ')!r}…")

    passed = True
    passed &= check("Response not empty", bool(text.strip()))
    passed &= check(f"Response length ≤ {tc.max_response_len}",
                    len(text) <= tc.max_response_len,
                    f"got {len(text)}")

    if tc.expected_zone_id is not None:
        passed &= check(f"Zone ID = {tc.expected_zone_id}",
                        params.get("location_id") == tc.expected_zone_id,
                        f"got {params.get('location_id')}")

    if tc.expected_hour is not None:
        passed &= check(f"Hour = {tc.expected_hour}",
                        params.get("hour") == tc.expected_hour,
                        f"got {params.get('hour')}")

    if tc.expected_dow is not None:
        passed &= check(f"DoW = {tc.expected_dow}",
                        params.get("day_of_week") == tc.expected_dow,
                        f"got {params.get('day_of_week')}")

    if tc.has_candidates:
        passed &= check("Has candidates (disambiguation)",
                        len(candidates) > 0, f"got {len(candidates)}")

    return passed


# ─── Test Definitions ─────────────────────────────────────────────────────────

TESTS: List[TestCase] = [

    # ── INTENT DETECTION ──────────────────────────────────────────────────────

    TestCase(
        name="INT-1 OOS greeting",
        message="Ciao! Come stai?",
        expected_intent="oos",
        max_response_len=250,
    ),
    TestCase(
        name="INT-2 OOS weather",
        message="Che tempo fa oggi a New York?",
        expected_intent="oos",
        max_response_len=250,
    ),
    TestCase(
        name="INT-3 PREDICT basic EN",
        message="How easy is it to find a taxi at JFK on Monday at 8am?",
        expected_zone_id=132,
        expected_hour=8,
        expected_dow=0,
    ),
    TestCase(
        name="INT-4 TREND pattern query",
        message="Di solito com'è la situazione a Times Square la sera?",
        expected_zone_id=230,
    ),
    TestCase(
        name="INT-5 TREND EN pattern query",
        message="What's the usual pattern at Midtown on Friday evenings?",
    ),

    # ── PARAMETER EXTRACTION ─────────────────────────────────────────────────

    TestCase(
        name="EXT-1 JFK Monday 8am IT",
        message="JFK lunedì alle 8 di mattina",
        expected_zone_id=132,
        expected_hour=8,
        expected_dow=0,
    ),
    TestCase(
        name="EXT-2 Times Square Sat 10pm EN",
        message="Times Square Saturday 10pm",
        expected_zone_id=230,
        expected_hour=22,
        expected_dow=5,
    ),
    TestCase(
        name="EXT-3 Numeric zone ID",
        message="Zona 236 mercoledì alle 9",
        expected_zone_id=236,
        expected_hour=9,
        expected_dow=2,
    ),
    TestCase(
        name="EXT-4 Hour range — pomeriggio",
        message="Come va Brooklyn il venerdì pomeriggio?",
        expected_zone_id=71,
        expected_dow=4,
    ),
    TestCase(
        name="EXT-5 Missing zone → ask_zone",
        message="Che situazione c'è lunedì alle 10?",
        max_response_len=250,
    ),
    TestCase(
        name="EXT-6 Ambiguous zone → candidates",
        message="Come si trova un taxi a Midtown?",
        has_candidates=False,   # Midtown may resolve or give candidates — accept both
    ),

    # ── MULTI-TURN CONTEXT ───────────────────────────────────────────────────

    TestCase(
        name="CTX-1 Follow-up day change",
        message="e domani?",
        # Simulate JFK Monday 8am already in session
        prev_params={"location_id": 132, "hour": 8, "day_of_week": 0, "month": 3},
        expected_zone_id=132,   # zone MUST be preserved
        expected_hour=8,        # hour MUST be preserved
    ),
    TestCase(
        name="CTX-2 Follow-up zone change",
        message="e a Brooklyn?",
        prev_params={"location_id": 132, "hour": 8, "day_of_week": 0, "month": 3},
        expected_zone_id=71,    # zone MUST update
        expected_hour=8,        # hour MUST be preserved
        expected_dow=0,         # dow MUST be preserved
    ),
    TestCase(
        name="CTX-3 Follow-up hour range",
        message="e la sera?",
        prev_params={"location_id": 132, "hour": 8, "day_of_week": 0, "month": 3},
        expected_zone_id=132,   # zone MUST be preserved
    ),

    # ── EDGE CASES ────────────────────────────────────────────────────────────

    TestCase(
        name="EDGE-1 Empty message robustness",
        message=".",
        max_response_len=400,
    ),
]

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 62)
    print("  NYC Taxi Agent v4 — Stress Test Suite (15 scenarios)")
    print("=" * 62)

    agent = get_agent()
    total, passed_count = len(TESTS), 0

    for tc in TESTS:
        ok = run_test(tc, agent)
        if ok:
            passed_count += 1

    print("\n" + "=" * 62)
    color = GREEN if passed_count == total else YELLOW if passed_count >= total * 0.8 else RED
    print(f"  Result: {color}{passed_count}/{total} tests passed{RESET}")
    print("=" * 62)

    # Save JSON report
    report_path = r'C:\Users\andre\Desktop\Progetto_Accenture\output\stress_test_results.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump({"passed": passed_count, "total": total}, f, indent=2)
    print(f"\n  Report saved: {report_path}")


if __name__ == "__main__":
    main()
