"""Command line: `forma pipeline | serve | eval`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="forma", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    pl = sub.add_parser("pipeline", help="download the sources and build the warehouse (bronze -> silver -> gold -> checks)")
    pl.add_argument("--refresh", action="store_true", help="re-download every source")
    sv = sub.add_parser("serve", help="run the dashboard and the agents")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8000)
    ev = sub.add_parser("eval", help="run the safety and quality scenarios")
    ev.add_argument("--json", help="also write the full report here")
    ev.add_argument("--llm", default="none", help="ollama:<model> to include the language model (default: off, deterministic)")
    args = p.parse_args(argv)
    if args.cmd == "pipeline":
        from .pipeline.run import build

        return 0 if build(refresh=args.refresh)["passed"] else 1
    if args.cmd == "serve":
        import uvicorn

        uvicorn.run("forma.api:app", host=args.host, port=args.port, log_level="info")
        return 0
    from .evaluate import run

    report = run(llm=args.llm)
    print(json.dumps(report["summary"], indent=2))
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2))
    return 0 if report["summary"]["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
