# Framework Integration

Alaya is intentionally framework-neutral. Integrations should preserve one
observable lifecycle instead of hiding memory updates inside a model callback:

```text
context -> activate experience -> record influenced decision
        -> observe outcome -> reinforce or contradict evidence
```

The runnable reference is [`examples/framework_adapter.py`](../examples/framework_adapter.py).
It has no framework dependency and can be used as the boundary around an agent
runner.

## Hook mapping

| Alaya boundary | LangGraph | OpenAI Agents SDK | CrewAI / AutoGen |
| --- | --- | --- | --- |
| `before_decision(context)` | node input or state preprocessor | instructions/context factory | task or message preprocessor |
| `after_decision(...)` | state update after agent node | run item / trace callback | task result callback |
| `after_outcome(...)` | terminal or evaluator node | application-owned evaluator | crew/chat completion evaluator |

These are lifecycle mappings, not bundled dependencies. Keep framework imports
in your adapter package and pass plain strings and Alaya models across the
boundary.

## Minimal pattern

```python
from examples.framework_adapter import AgentExperienceAdapter

adapter = AgentExperienceAdapter(existing_seeds)
guidance = adapter.before_decision(task_context)

# Render bounded guidance into your framework's context, then run the agent.
decision = adapter.after_decision(
    task_context,
    [item["seed_id"] for item in guidance],
    action="produce reviewed implementation",
)

# Outcome polarity comes from an external evaluator or operator, not the agent
# that produced the action.
adapter.after_outcome(
    decision,
    outcome="acceptance tests passed",
    polarity="success",
    source_id="ci-run-184",
)
```

## Required safeguards

- Keep the IDs of experience seeds that actually influenced a decision.
- Obtain outcome polarity from an independent test, reviewer, or operator.
- Use stable source IDs so retries do not duplicate evidence.
- Bound the number and size of activated lessons before prompt construction.
- Do not store secrets, raw private transcripts, protected traits, or covert
  psychological profiles as experience seeds.
- Keep high-stakes decisions under qualified human review.

## Conformance check

Run the framework-neutral adapter and the memory conformance evaluation:

```bash
python examples/framework_adapter.py
python evals/memory_conformance_eval.py
```

An integration is useful only if a maintainer can inspect why a seed was
activated, which decision it influenced, and which independent outcome later
changed it.
