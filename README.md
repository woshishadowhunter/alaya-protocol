# Alaya Protocol

[English](README.md) | [简体中文](README.zh-CN.md)

**Give agents experience, not just memory.**

Alaya is a zero-dependency Python layer that turns observed outcomes into
auditable, revisable experience seeds. An agent can retrieve relevant lessons
before a decision, but a single LLM reflection never becomes behavioral truth
by itself.

[![CI](https://github.com/woshishadowhunter/alaya-protocol/actions/workflows/ci.yml/badge.svg)](https://github.com/woshishadowhunter/alaya-protocol/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/woshishadowhunter/alaya-protocol)](https://github.com/woshishadowhunter/alaya-protocol/releases/latest)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg)](pyproject.toml)
[![License: Apache--2.0](https://img.shields.io/badge/License-Apache--2.0-green.svg)](LICENSE)

> Alaya's seed/activation vocabulary is inspired by Yogacara philosophy. The implementation is an engineering model, not a claim that software is conscious or that Buddhist philosophy has been computationally reproduced.

## Why

Most agent memory answers *what happened*. Alaya captures a narrower object:

```text
situation -> action -> observed outcome -> candidate lesson
          -> independent evidence -> bounded behavioral guidance
```

Facts belong in knowledge stores, preferences in consented user profiles, and explicit rules in project instructions. Alaya is for contextual practical judgment.

## Install

```bash
python -m pip install https://github.com/woshishadowhunter/alaya-protocol/releases/download/v0.3.1/alaya_protocol-0.3.1-py3-none-any.whl
alaya --help
```

Alaya requires Python 3.10+ and has no runtime dependencies outside the standard library.

## Five-minute example

Plant a lesson from one outcome:

```bash
alaya plant \
  --lesson "Align stakeholders before full solution design" \
  --guidance "Start with stakeholder mapping and a resistance check" \
  --tags "project,stakeholder,community" \
  --applies "multi-party projects" \
  --source "retrospective-001" \
  --evidence "A complete design stalled before interests were aligned"
```

The seed remains a `candidate`. Add independent evidence with the returned ID:

```bash
alaya reinforce SEED_ID --polarity support --source retrospective-002 \
  --evidence "An alignment workshop unlocked delivery"
```

Retrieve relevant active experience:

```bash
alaya activate "planning a community project with multiple property owners"
```

Every result includes its relevance, confidence, recency, matched terms, evidence, applicability, and counterexamples.

## Python API

```python
from alaya import Evidence, ExperienceEngine, ExperienceSeed

seed = ExperienceSeed.new(
    lesson="Align stakeholders before full solution design.",
    guidance="Map interests and resistance first.",
    context_tags=["project", "stakeholder"],
    applicability="Multi-party projects",
    evidence=Evidence("support", "case-1", "Early design stalled"),
)

engine = ExperienceEngine()
seed = engine.reinforce(seed, Evidence("support", "case-2", "Alignment unlocked delivery"))
matches = engine.activate("A stakeholder-heavy project", [seed])
```

## What v0.3 includes

- Portable JSON Schema at `protocol/experience-seed.schema.json`
- Immutable Python models and standard-library SQLite storage
- Deterministic promotion, contradiction, decay, and activation policy
- Explainable retrieval rather than hidden profile changes
- `learn-from-experience` Skill for ChatGPT/Codex-compatible environments
- A community project decision demo and deterministic evaluation
- A framework adapter pattern and conformance benchmark for integrations

See the [framework integration guide](docs/integrations.md) for lifecycle hooks,
safeguards, and mappings for LangGraph, OpenAI Agents SDK, CrewAI, and AutoGen.

## Privacy and safety

Storage is local by default and no network calls are made. Do not store raw private transcripts, secrets, protected traits, medical details, or covert psychological profiles. Keep legal, medical, financial, employment, eligibility, and other high-stakes decisions under qualified human review. Inspect, export, or delete every seed through the CLI.

## Roadmap

1. v0.1 - individual experience lifecycle and Skill
2. v0.2 - pluggable retrieval, counterexamples, and scalable storage
3. v0.3 - evidence natures, channel weighting, feedback, rules, and self-audit
4. Next - longitudinal evaluation and production framework adapters

See [architecture](docs/architecture.md), [use cases](docs/use-cases.md), [comparison and benchmark notes](docs/comparison-and-benchmark.md), [contributing](CONTRIBUTING.md), and the [application readiness plan](docs/codex-for-oss-application.md).

## License

Apache License 2.0.
