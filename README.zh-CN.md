# Alaya Protocol

[English](README.md) | [简体中文](README.zh-CN.md)

**给长期运行的 AI Agent 增加“可审计经验”，而不只是记忆。**

Alaya Protocol 是一个轻量、框架中立的 Python 协议层。它把重复出现的结果、反例和证据沉淀为可检查、可修订、可删除的 experience seeds。Agent 可以在做决策前检索相关经验，但单次 LLM 反思不会直接变成不可质疑的行为规则。

> Alaya 的 seed / activation 词汇受到唯识学启发，但本项目是工程模型，不声称软件具有意识，也不声称复现了佛教哲学。

## 它解决什么问题

很多 agent memory 系统记录的是“发生过什么”。这对长期行为改进还不够，因为：

- 一次失败总结可能是错的。
- 记忆可能混入偏见、隐私或未经验证的个人画像。
- Agent 可能把单次反思当作长期规则。
- 用户和维护者难以审计“为什么这条经验影响了当前决策”。
- 多轮项目中，经验需要保留反例、适用边界和证据来源。

Alaya 关注更窄但更安全的对象：

```text
situation -> action -> observed outcome -> candidate lesson
          -> independent evidence -> bounded behavioral guidance
```

## 核心理念

- 事实应该放进知识库。
- 用户偏好应该放进有明确同意边界的用户档案。
- 明确规则应该放进项目指令。
- Alaya 只处理“带证据和适用边界的实践经验”。

## 快速安装

```bash
python -m pip install -e .
alaya --help
```

Alaya 需要 Python 3.10+，运行时只依赖标准库。

## 五分钟示例

写入一条候选经验：

```bash
alaya plant \
  --lesson "Align stakeholders before full solution design" \
  --guidance "Start with stakeholder mapping and a resistance check" \
  --tags "project,stakeholder,community" \
  --applies "multi-party projects" \
  --source "retrospective-001" \
  --evidence "A complete design stalled before interests were aligned"
```

添加独立支持证据：

```bash
alaya reinforce SEED_ID --polarity support --source retrospective-002 \
  --evidence "An alignment workshop unlocked delivery"
```

检索相关经验：

```bash
alaya activate "planning a community project with multiple property owners"
```

输出会包含相关度、置信度、时效、匹配词、证据、适用边界和反例。

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

## 当前能力

- 可移植 JSON Schema：`protocol/experience-seed.schema.json`。
- 不可变 Python 模型和标准库 SQLite 存储。
- 确定性的提升、矛盾、衰减和激活策略。
- 可解释检索，而不是隐藏式个人画像变化。
- 适用于 ChatGPT/Codex 环境的 `learn-from-experience` Skill。
- 社区项目决策 demo 和确定性评估。

## 隐私和安全边界

默认本地存储，不发起网络请求。不要存储原始私人对话、密钥、受保护特征、医疗细节或隐蔽心理画像。法律、医疗、金融、就业、资格判断等高风险场景必须由合格人类审核。每条 seed 都应该可以检查、导出或删除。

## 适用场景

- 长期运行的 coding agent。
- 项目复盘和经验沉淀。
- 多 agent 协作中的经验证据层。
- 本地优先的 agent memory 实验。
- 需要解释“为什么这条经验影响了当前建议”的 agent 系统。

更多场景见 [docs/use-cases.md](docs/use-cases.md)。

## 与 Codex for OSS 的关系

Alaya 可以帮助维护者研究“agent 如何从项目经验中学习而不失控”。Codex/API credits 最适合用于：

- 评估候选经验抽取质量。
- 辅助 issue triage 和 PR review。
- 维护框架适配器和文档。
- 对真实 agent trace 做可选、受控的经验提取实验。

确定性策略、schema、测试和本地存储仍然保持模型无关。

## 路线图

1. v0.1 - 单 agent experience lifecycle 和 Skill。
2. v0.2 - 可插拔语义检索和框架适配器。
3. v0.3 - agent 间关系经验和作用域信任证据。
4. v0.4 - 多 agent 协商和长期评估。

## 开发

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

## 贡献

欢迎贡献确定性评估、协议示例、文档翻译、隐私/安全审查和框架适配器。请先阅读 [CONTRIBUTING.md](CONTRIBUTING.md)。

## License

Apache License 2.0.

