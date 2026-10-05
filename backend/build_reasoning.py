"""Fill cache/reasoning/<slug>.json by running Nemotron once per bundled case.

    python build_reasoning.py [slug ...]
This spends Token Factory credit. The running app never does it for a bundled case.
"""
import json, sys
from pathlib import Path
import cases, reasoning

HERE = Path(__file__).resolve().parent
for slug in (sys.argv[1:] or [c["slug"] for c in cases.CASES]):
    rec = json.loads((HERE / "cache" / "cases" / slug / "record.json").read_text())
    out = reasoning.reason(rec, cases.BY_SLUG[slug]["procedure"])
    reasoning.save(slug, out)
    cp = out["checkpoint"]
    print(slug, cp["decision"], cp.get("at_s"), "|", cp["headline"],
          "| grounded", sum(e["verified"] for e in cp["evidence"]), "/", len(cp["evidence"]),
          out["calls"], flush=True)
