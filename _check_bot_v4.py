"""
AST + import check per Agent v4 + Groq integration.
Non richiede Ollama, modelli ML o variabili d'ambiente reali.
"""
import ast
import pathlib

BASE = pathlib.Path(__file__).parent
PASS_SYM = "\033[92m[OK]\033[0m"
FAIL_SYM = "\033[91m[FAIL]\033[0m"
errors = []

def ok(msg):   print(f"  {PASS_SYM} {msg}")
def fail(msg): print(f"  {FAIL_SYM} {msg}"); errors.append(msg)

def get_src(rel): return (BASE / rel).read_text(encoding="utf-8")
def parse(rel):   return ast.parse(get_src(rel))

# ── 1. config.py: Groq vars presenti ─────────────────────────────────────────
cfg = get_src("llm_tool/config.py")
for var in ("GROQ_API_KEY", "GROQ_MODEL", "LLM_PROVIDER"):
    (ok if var in cfg else fail)(f"config.py: {var} presente")

# ── 2. llm_factory.py: esiste e ha get_llm ───────────────────────────────────
fac_path = BASE / "llm_tool" / "llm_factory.py"
if fac_path.exists():
    ok("llm_factory.py: file creato")
    fac = get_src("llm_tool/llm_factory.py")
    tree_f = ast.parse(fac)
    fns = [n.name for n in ast.walk(tree_f) if isinstance(n, ast.FunctionDef)]
    (ok if "get_llm" in fns else fail)("llm_factory.py: get_llm() presente")
    (ok if "_create_groq" in fns else fail)("llm_factory.py: _create_groq() presente")
    (ok if "_create_ollama" in fns else fail)("llm_factory.py: _create_ollama() presente")
    (ok if "ChatGroq" in fac else fail)("llm_factory.py: ChatGroq referenziato")
    (ok if "ChatOllama" in fac else fail)("llm_factory.py: ChatOllama (fallback) presente")
else:
    fail("llm_factory.py: FILE NON TROVATO")

# ── 3. input_validator.py: factory + override ────────────────────────────────
val = get_src("llm_tool/input_validator.py")
tree_v = ast.parse(val)
methods_v = [n.name for n in ast.walk(tree_v) if isinstance(n, ast.FunctionDef)]

(ok if "ChatOllama" not in val else fail)("input_validator.py: ChatOllama diretto rimosso")
(ok if "get_llm" in val else fail)("input_validator.py: get_llm() importato")
(ok if "_deterministic_override" in methods_v else fail)("input_validator.py: _deterministic_override presente")
(ok if "_deterministic_override" in val and "_sanitize_extracted" in val else fail)(
    "input_validator.py: override chiamato dopo sanitize")

# ── 4. agent.py: factory + no ChatOllama diretto ─────────────────────────────
agt = get_src("llm_tool/agent.py")
(ok if "ChatOllama" not in agt else fail)("agent.py: ChatOllama diretto rimosso")
(ok if "get_llm" in agt else fail)("agent.py: get_llm() importato/usato")
(ok if "LLM_MODEL" not in agt or "LLM_BASE_URL" not in agt else fail)(
    "agent.py: LLM_MODEL/LLM_BASE_URL rimossi (gestiti dalla factory)")

# ── 5. AgentState: hour_range presente, language assente ─────────────────────
tree_a = ast.parse(agt)
state_cls = next(n for n in ast.walk(tree_a) if isinstance(n, ast.ClassDef) and n.name == "AgentState")
keys = [t.target.id for t in state_cls.body if isinstance(t, ast.AnnAssign) and hasattr(t.target, "id")]
(ok if "hour_range" in keys else fail)(f"AgentState: hour_range presente (keys={keys})")
(ok if "language" not in keys else fail)("AgentState: language assente")

# ── 6. .env: GROQ_API_KEY presente ───────────────────────────────────────────
env = get_src(".env")
(ok if "GROQ_API_KEY" in env else fail)(".env: GROQ_API_KEY presente")
(ok if "LLM_PROVIDER" in env else fail)(".env: LLM_PROVIDER presente")

# ── 7. langchain-groq installato ─────────────────────────────────────────────
try:
    import importlib
    importlib.import_module("langchain_groq")
    ok("langchain-groq: pacchetto installato")
except ImportError:
    fail("langchain-groq: NON installato — eseguire: pip install langchain-groq")

# ── Summary ───────────────────────────────────────────────────────────────────
print()
if not errors:
    print("=== TUTTI I CHECK SUPERATI [PASS] ===")
else:
    print(f"=== {len(errors)} CHECK FALLITI ===")
    for e in errors:
        print(f"  - {e}")
