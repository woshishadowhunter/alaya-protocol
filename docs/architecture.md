# Architecture

## Design boundary

Alaya does not choose actions or call a language model. An application may ask an LLM to propose a lesson, but Alaya validates, stores, scores, retrieves, and explains that proposal through deterministic policy.

## Components

- `models.py`: immutable experience and evidence types.
- `engine.py`: promotion, contradiction, decay, and lexical activation.
- `store.py`: local SQLite persistence and portable JSON export.
- `cli.py`: stable human and Skill-facing command surface.
- `protocol/`: language-neutral JSON contract.
- `skills/`: first installable agent workflow.

## Trust model

Candidate extraction is untrusted. Evidence source IDs prevent one event from being counted twice. Activation requires active status and contextual overlap. Results expose scoring factors. Contradictions remain attached to the seed. Callers own consent, redaction, and data-retention policy.

## Yogacara inspiration

The engineering analogy is deliberately limited: observations resemble perception; extracted contextual lessons resemble seeds; repeated evidence resembles reinforcement; context retrieval resembles activation. Alaya does not model the eight consciousnesses as literal software entities and makes no metaphysical or scientific claim about machine consciousness.

