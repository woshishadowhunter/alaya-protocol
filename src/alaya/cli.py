from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from .engine import EvolutionPolicy, ExperienceEngine
from .models import Channel, Decision, Evidence, ExperienceSeed, Nature, Observation, RuleType
from .store import DecisionStore, SQLiteSeedStore


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
    plant.add_argument("--channel", choices=["direct", "reflective", "interactive"], default="reflective")
    plant.add_argument("--nature", choices=["speculative", "conditional", "principle"], default="speculative")

    reinforce = commands.add_parser("reinforce", help="Add independent evidence")
    reinforce.add_argument("seed_id")
    reinforce.add_argument("--polarity", choices=["support", "contradict"], required=True)
    reinforce.add_argument("--source", required=True)
    reinforce.add_argument("--evidence", required=True)
    reinforce.add_argument("--channel", choices=["direct", "reflective", "interactive"], default="reflective")

    activate = commands.add_parser("activate", help="Find experience relevant to current context")
    activate.add_argument("context"); activate.add_argument("--limit", type=int, default=5)

    apply_cmd = commands.add_parser("apply", help="Record a decision influenced by experience")
    apply_cmd.add_argument("context")
    apply_cmd.add_argument("--action", required=True)
    apply_cmd.add_argument("--choose", required=True, help="Comma-separated seed IDs to adopt")
    apply_cmd.add_argument("--limit", type=int, default=5)

    observe_cmd = commands.add_parser("observe", help="Observe outcome and auto-reinforce seeds")
    observe_cmd.add_argument("decision_id")
    observe_cmd.add_argument("--outcome", required=True)
    observe_cmd.add_argument("--polarity", choices=["success", "failure", "mixed"], required=True)
    observe_cmd.add_argument("--source", required=True)

    rule_cmd = commands.add_parser("rule", help="Harden or check inference rules")
    rule_sub = rule_cmd.add_subparsers(dest="rule_command", required=True)
    rule_harden = rule_sub.add_parser("harden", help="Create a hardened inference rule")
    rule_harden.add_argument("--lesson", required=True)
    rule_harden.add_argument("--guidance", required=True)
    rule_harden.add_argument("--type", choices=["correction", "heuristic", "boundary", "pattern"], required=True)
    rule_harden.add_argument("--triggers", required=True, help="Comma-separated trigger tokens")
    rule_harden.add_argument("--applies", required=True)
    rule_harden.add_argument("--source")
    rule_harden.add_argument("--evidence")
    rule_harden.add_argument("--channel", choices=["direct", "reflective", "interactive"], default="reflective")
    rule_check = rule_sub.add_parser("check", help="Check context against rules")
    rule_check.add_argument("context")

    audit_cmd = commands.add_parser("audit", help="Run cognitive self-audit")
    audit_sub = audit_cmd.add_subparsers(dest="audit_command", required=True)
    audit_sub.add_parser("health", help="Cognitive health check")
    audit_blind = audit_sub.add_parser("blindspots", help="Detect uncovered domains")
    audit_blind.add_argument("--domains", help="Comma-separated known domains to check coverage")

    commands.add_parser("list", help="List seeds")
    show = commands.add_parser("show", help="Show one seed"); show.add_argument("seed_id")
    commands.add_parser("export", help="Export portable JSON")
    delete = commands.add_parser("delete", help="Delete a seed"); delete.add_argument("seed_id")
    return root


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    store = SQLiteSeedStore(Path(args.db))
    dstore = DecisionStore(Path(args.db))
    engine = ExperienceEngine()
    now = datetime.now(timezone.utc)

    if args.command == "plant":
        evidence = None
        if bool(args.source) != bool(args.evidence):
            raise SystemExit("--source and --evidence must be supplied together")
        if args.source:
            evidence = Evidence("support", args.source, args.evidence, now, channel=args.channel)
        seed = ExperienceSeed.new(
            lesson=args.lesson, guidance=args.guidance,
            context_tags=[tag.strip() for tag in args.tags.split(",")],
            applicability=args.applies, nature=args.nature, evidence=evidence, now=now,
        )
        store.save(seed); _print(seed.to_dict()); return 0
    if args.command == "reinforce":
        seed = _required(store, args.seed_id)
        evolved = engine.reinforce(seed, Evidence(args.polarity, args.source, args.evidence, now, channel=args.channel), now)
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
    if args.command == "apply":
        chosen = [cid.strip() for cid in args.choose.split(",") if cid.strip()]
        seeds = store.list_active()
        decision, activations = engine.apply(args.context, seeds, chosen, args.action, now, args.limit)
        dstore.save(decision)
        for a in activations:
            if a.seed.id in chosen:
                store.save(a.seed.with_changes(last_activated_at=now))
        _print({
            "decision": decision.to_dict(),
            "activations": [{"seed": a.seed.to_dict(), "score": a.score, "explanation": a.explanation} for a in activations],
        }); return 0
    if args.command == "observe":
        decision = dstore.get(args.decision_id)
        if decision is None:
            raise SystemExit(f"decision not found: {args.decision_id}")
        observation = Observation(args.decision_id, args.outcome, args.polarity, args.source, now)
        seeds = store.list_active(since_days=3650)
        evolved = engine.observe(decision, observation, seeds, now)
        for seed in evolved:
            store.save(seed)
        _print([seed.to_dict() for seed in evolved]); return 0
    if args.command == "rule":
        if args.rule_command == "harden":
            evidence = None
            if args.source and args.evidence:
                evidence = Evidence("support", args.source, args.evidence, now, channel=args.channel)
            seed = ExperienceSeed.new_rule(
                lesson=args.lesson, guidance=args.guidance,
                rule_type=args.type,
                trigger_tokens=[t.strip() for t in args.triggers.split(",")],
                applicability=args.applies, evidence=evidence, now=now,
            )
            store.save(seed); _print(seed.to_dict()); return 0
        if args.rule_command == "check":
            seeds = store.list_active()
            rules = [s for s in seeds if s.rule_type is not None]
            results = engine.check_rules(args.context, rules)
            _print([{"rule": r[0].to_dict(), "relevance": r[1], "breach": r[2]} for r in results])
            return 0
    if args.command == "audit":
        if args.audit_command == "health":
            seeds = store.list()
            result = engine.audit_health(seeds)
            _print(result); return 0
        if args.audit_command == "blindspots":
            seeds = store.list()
            domains = args.domains.split(",") if getattr(args, "domains", None) else None
            result = engine.audit_blindspots(seeds, domains)
            _print(result); return 0
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

