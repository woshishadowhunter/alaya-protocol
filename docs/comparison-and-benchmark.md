# Comparison And Benchmark Notes

Alaya should not compete as another vector memory database. Its sharper role is
to define a small, auditable experience layer that other agent frameworks can
call before and after decisions.

## Positioning

| Category | Typical value | Alaya boundary |
| --- | --- | --- |
| Vector memory | Recall semantically similar text | Stores bounded lessons with evidence, status, confidence, and counterexamples |
| Agent runtime | Plans and executes tool calls | Provides pre-decision guidance and post-outcome feedback |
| Long-term profile | Remembers user preferences | Avoids covert profiling; every seed is inspectable and deletable |
| Rule engine | Enforces explicit policies | Evolves practical lessons only after evidence and contradictions |

## Conformance Checks

Run:

```bash
PYTHONPATH=src python evals/memory_conformance_eval.py
```

The current deterministic benchmark checks three minimum behaviors:

- Relevant experience activates with an explanation and matched terms.
- A chosen seed can be reinforced by a later successful outcome.
- Contradictory evidence lowers confidence instead of being ignored.

An adapter should pass these behaviors before it is described as
Alaya-compatible.

## Adapter Roadmap

The recommended adapter shape is shown in `examples/framework_adapter.py`:

1. `before_decision(context)` returns ranked, explainable guidance.
2. `after_decision(context, chosen_seed_ids, action)` records the actual choice.
3. `after_outcome(decision, outcome, polarity, source_id)` updates evidence.

This keeps Alaya framework-neutral while making integrations with LangGraph,
OpenAI Agents SDK, AutoGen, CrewAI, or custom runners straightforward.

