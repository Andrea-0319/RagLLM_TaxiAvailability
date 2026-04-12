"""Test InputValidator parameter extraction."""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, ".")

from llm_tool import get_validator

validator = get_validator()

queries = [
    "Quanti taxi a Midtown lunedi alle 8 a marzo?",
    "Times Square sabato sera alle 17:30",
    "JFK venerdi a mezzanotte",
    "Brooklyn il 15 febbraio alle 14",
]

for i, q in enumerate(queries, 1):
    print(f"\n{'='*60}")
    print(f"Query {i}: {q}")
    print("="*60)
    params = validator.extract(q)
    print(f"Extracted params: {params}")