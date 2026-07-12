---
name: learn-from-experience
description: Turn repeated work or life outcomes into auditable, revisable experience seeds and apply relevant prior lessons to new decisions. Use when a user asks an agent to learn from an outcome, remember a practical lesson, conduct a retrospective, avoid repeating a mistake, apply prior experience, explain which experiences influenced advice, or revise an earlier lesson after counterevidence.
---

# Learn from Experience

Build practical judgment from evidence without treating one reflection as permanent truth.

## Workflow

1. Before advising on a recurring decision, run `scripts/alaya_skill.py activate --context "<current situation>"`.
2. Use only returned active seeds that fit the stated applicability. Cite the lesson and its limits; never hide that prior experience influenced advice.
3. After an outcome, distinguish facts, user preference, and experience. Draft one candidate: situation, action, outcome, lesson, future guidance, applicability, counterexample.
4. Ask for confirmation before storing sensitive or identity-shaping experience unless the user explicitly requested storage.
5. Plant the candidate with a unique source ID. It cannot guide behavior until independent support promotes it.
6. Reinforce later supporting or contradicting outcomes with new source IDs. Preserve contradictions.
7. On request, list, show, export, or delete seeds. Treat deletion as final.

## Commands

```bash
python scripts/alaya_skill.py plant --lesson "..." --guidance "..." --tags "project,stakeholder" --applies "..." --source "case-id" --evidence "..."
python scripts/alaya_skill.py reinforce SEED_ID --polarity support --source "new-case-id" --evidence "..."
python scripts/alaya_skill.py activate --context "..."
python scripts/alaya_skill.py list
python scripts/alaya_skill.py show SEED_ID
python scripts/alaya_skill.py export
python scripts/alaya_skill.py delete SEED_ID
```

Default storage is `.alaya/seeds.db`. Pass `--db PATH` before the command to choose another local store.

## Guardrails

- Treat model-generated lessons as proposals, not observations.
- Require outcome evidence; do not infer lessons from tone or identity.
- Never create covert psychological profiles or store raw private transcripts.
- Avoid secrets, medical details, protected traits, and high-stakes autonomous decisions.
- Prefer narrow applicability and record counterexamples.
- Never convert a seed into an irreversible instruction.

Read [references/experience-model.md](references/experience-model.md) when distinguishing experience from facts, preferences, or rules.

