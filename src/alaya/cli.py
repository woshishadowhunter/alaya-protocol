from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from .engine import EvolutionPolicy, ExperienceEngine
from .models import Evidence, ExperienceSeed, Nature
from .store import SQLiteSeedStore


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="alaya", description="Auditable experience evolution for AI agents")
    root.add_argument("--db", default=".alaya/seeds.db", help="SQLite seed database")
    commands = root.add_subparsers(dest="command", required=True)

    plant = commands.add_parser("plant", help="Plant a candidate experience seed")
    plant.add_argument("--lesson", required=True)
    plant.add_argument("--guidance", required=True)
    plant.add_argument("--tags", required=True)
    plant.add_argument("--applies", required=True)
    plant.add_argument("--source")
    plant.add_argument("--evidence")
    plant.add_argument("--nature", choices=["speculative", "conditional", "principle"], default="speculative")

    reinforce = commands.add_parser("reinforce", help="Add independent evidence")
    reinforce.add_argument("seed_id"); reinforce.add_argument("--polarity", choices=["support", "contradict"], required=True)
    reinforce.add_argument("--source", required=True); reinforce.add_argument("--evidence", required=True)

    activate = commands.add_parser("activate", help="Find experience relevant to current context")
    activate.add_argument("context"); activate.add_argument("--limit", type=int, default=5)

    commands.add_parser("list", help="List seeds")
    show = commands.add_parser("show", help="Show one seed"); show.add_argument("seed_id")
    commands.add_parser("export", help="Export portable JSON")
    delete = commands.add_parser("delete", help="Delete a seed"); delete.add_argument("seed_id")
    return root


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    store = SQLiteSeedStore(Path(args.db))
    engine = ExperienceEngine()
    now = datetime.now(timezone.utc)

    if args.command == "plant":
        evidence = None
        if bool(args.source) != bool(args.evidence):
            raise SystemExit("--source and --evidence must be supplied together")
        if args.source:
            evidence = Evidence("support", args.source, args.evidence, now)
        seed = ExperienceSeed.new(
            lesson=args.lesson, guidance=args.guidance,
            context_tags=[tag.strip() for tag in args.tags.split(",")],
            applicability=args.applies, nature=args.nature, evidence=evidence, now=now,
        )
        store.save(seed); _print(seed.to_dict()); return 0
    if args.command == "reinforce":
        seed = _required(store, args.seed_id)
        evolved = engine.reinforce(seed, Evidence(args.polarity, args.source, args.evidence, now), now)
        store.save(evolved); _print(evolved.to_dict()); return 0
    if args.command == "activate":
        items = engine.activate(args.context, store.list_active(), now, args.limit)
        _print([{"seed": item.seed.to_dict(), "score": item.score, "relevance": item.relevance,
                 "confidence": item.confidence, "recency": item.recency,
                 "matched_terms": list(item.matched_terms),
                 "explanation": item.explanation} for item in items]); return 0
    if args.command == "list":
        _print([seed.to_dict() for seed in store.list_active(since_days=3650)]); return 0
    if args.command == "show":
        _print(_required(store, args.seed_id).to_dict()); return 0
    if args.command == "export":
        print(store.export_json()); return 0
    if args.command == "delete":
        if not store.delete(args.seed_id):
            raise SystemExit(f"seed not found: {args.seed_id}")
        _print({"deleted": args.seed_id}); return 0
    return 2


def _required(store: SQLiteSeedStore, seed_id: str) -> ExperienceSeed:
    seed = store.get(seed_id)
    if seed is None:
        raise SystemExit(f"seed not found: {seed_id}")
    return seed


def _print(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())

