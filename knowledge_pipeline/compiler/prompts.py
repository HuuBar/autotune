"""M4 编译提示词模板（选卡 + 成图两步，中文）。

语义来源：01 文档 §3（编译器职责 ①~④）与 03 文档第一部分（S4a/S4b 质量门）。
输出格式硬约束：

* 选卡（S4a）：输出**纯 JSON** 论证表，每张卡 ``{card_id, decision, rationale}``，
  ``decision ∈ {adopt, reject}``，adopt 须答"为何适用于本项目"，reject 须有理由；
  论证表必须覆盖卡库全部卡 id。
* 成图（S4b）：输出**纯 YAML** 编排图，顶层键 / 节点字段 / 边类型严格遵守
  SPEC §1 schema（提示词内嵌清单，产物的确定性把关由 M1 校验器执行，不靠提示词）。

两步 prompt 都是输入的纯函数（无日期/随机量），保证 CachedClient 按内容寻址可复现。
阶段标记 ``【阶段：选卡】`` / ``【阶段：成图】`` 同时供 FakeClient 注册匹配。
"""

from __future__ import annotations

import json
from typing import Any, Iterable, Mapping

__all__ = [
    "CARD_SELECTION_SYSTEM",
    "GRAPH_SYSTEM",
    "card_selection_user",
    "graph_user",
]

#: 阶段标记（FakeClient 匹配锚点）
CARD_SELECTION_MARKER = "【阶段：选卡】"
GRAPH_MARKER = "【阶段：成图】"

CARD_SELECTION_SYSTEM = f"""{CARD_SELECTION_MARKER}
你是自动化 ML 优化系统的编排编译器，现在执行第一步：选卡论证（S4a）。

你会收到：① 战术知识卡库（Markdown，每卡含 情境/做法/依据/边界/验证方法/落点）；
② 任务书与地形描述（YAML，含任务目标、验收断言、基线指标、已知地形与待侦查项）。

你的职责：对卡库中**每一张卡**做出采用（adopt）或拒绝（reject）决策并给出论证：
- adopt：必须回答"为何适用于本项目"——引用任务书基线/地形事实说明适配点，并指出
  它将实例化为图中的哪类结构（领域/假说/动作约束/决策分支/写回纪律）；
- reject：必须给出理由（不适用本任务、超出本轮范围、由其他层承担、与采用卡重复等）。
这是防盲抄的审计点：不允许无理由采用，也不允许整批拒绝。

【输出格式硬约束】
- 只输出一个 JSON 数组，不要输出任何其他文字、解释或 Markdown 代码围栏；
- 数组元素：{{"card_id": "<卡id>", "decision": "adopt"|"reject", "rationale": "<非空论证>"}}；
- 数组必须覆盖卡库中出现的全部卡 id，不多不少，card_id 逐字照抄（如 "C2"、"B7"）。
"""

GRAPH_SYSTEM = f"""{GRAPH_MARKER}
你是自动化 ML 优化系统的编排编译器，现在执行第二步：成图（S4b）。

你会收到：① 战术知识卡库；② 任务书与地形描述；③ 第一步的选卡论证表。
你的职责：把采用卡的方法论实例化为一张四层编排图（L0 目标 / L1 领域 / L2 假说 /
L3 动作），并标注每个假说的先验（prior）与验证成本（cost）。

【结构要求】
- L0 目标：顶层 goal 字段，statement + acceptance（验收断言逐条机器可判：
  "<metric_path> <op> <number>"，op ∈ >= <= == > <）；
- L1 领域层：≥3 个正交方向，每个 domain 必须有非空 rationale（来源卡/任务书依据）；
- L2 假说层：每个 hypothesis 有 domain 引用、非空 mechanism（机制解释）、
  prior ∈ [0,1]、expected {{metric, delta, confidence}}、cost {{recon_steps, run_minutes}}；
- L3 动作层：每个 action 有 belongs_to（hypothesis 或 domain）、type ∈ recon|edit|run、
  do、tool_hint、produces（产物名）、非空 verify（成功判据）、retry_budget >= 0；
- 边：type ∈ requires|informs|alternatives|no_parallel；requires 可指向 action 或
  decision（决策点等待输入产物），不得来自 decision；全图不得有环；
- 决策点 decisions[]：{{id, after: <action id>, read: <该 action 的 produces>,
  branches}}；if 分支须带非空 then；else 分支的 else 值本身即兜底动作描述；
  分支须完备（存在 else 或分支数 >= 2）；
- 顶层键齐全：version / meta（task, created_by, source_cards=采用卡id列表）/
  goal / nodes / edges / decisions / writeback（ledger, record_schema, after_domain_review）。

【输出格式硬约束】
- 只输出编排图的 YAML 全文，不要输出任何其他文字或解释；
- 可以用 ```yaml 代码围栏包裹，围栏内必须是合法 YAML；
- 不得绕过上述结构要求——产出将经过确定性校验器，不过即返工。
"""


def _brief_section(brief_text: str) -> str:
    return f"=== 任务书与地形描述（YAML） ===\n{brief_text.strip()}\n"


def _cards_section(cards_text: str) -> str:
    return f"=== 战术知识卡库（Markdown） ===\n{cards_text.strip()}\n"


def _feedback_section(errors: Iterable[str] | None) -> str:
    """校验/自检失败回灌段：把上一次失败的错误列表明确交给 LLM 修正。"""
    items = [e for e in (errors or []) if str(e).strip()]
    if not items:
        return ""
    lines = "\n".join(f"  {i}. {e}" for i, e in enumerate(items, 1))
    return (
        "=== 上一次输出未通过（必须逐条修正后重新输出完整结果） ===\n"
        f"{lines}\n"
    )


def card_selection_user(
    cards_text: str,
    brief_text: str,
    *,
    errors: Iterable[str] | None = None,
) -> str:
    """选卡步 user prompt。``errors`` 为上一次失败的解析/自检问题（重试时回灌）。"""
    return (
        f"{CARD_SELECTION_MARKER}\n"
        + _cards_section(cards_text)
        + _brief_section(brief_text)
        + _feedback_section(errors)
        + "请输出采用/拒绝论证表（纯 JSON 数组，覆盖全部卡 id）。\n"
    )


def graph_user(
    cards_text: str,
    brief_text: str,
    card_decisions: list[Mapping[str, Any]],
    *,
    errors: Iterable[str] | None = None,
) -> str:
    """成图步 user prompt。``errors`` 为校验门/自检错误列表（重试时回灌）。"""
    decisions_json = json.dumps(
        list(card_decisions), ensure_ascii=False, indent=2
    )
    return (
        f"{GRAPH_MARKER}\n"
        + _cards_section(cards_text)
        + _brief_section(brief_text)
        + f"=== 选卡论证表（S4a 产物，成图须与其一致：source_cards = 全部 adopt 卡） ===\n{decisions_json}\n"
        + _feedback_section(errors)
        + "请输出编排图（纯 YAML，可用 ```yaml 围栏）。\n"
    )
