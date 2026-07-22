"""M4 编排编译器：选卡论证 → 成图 → 确定性校验门（不绕过校验）。

流程（SPEC §6 / 01 文档 §3 / 03 文档 S4a-S4b）：

1. **选卡**（S4a）：LLM 输出采用/拒绝论证表（JSON）。解析失败或自检不过
   （未覆盖全卡库 / 缺 rationale）按一次失败处理，回灌问题重试，≤ max_retries。
2. **成图**（S4b）：LLM 输出编排图 YAML。鲁棒解析（容忍 ```yaml 围栏与首尾杂讯）。
3. **校验门**：产出图过 ``validator``（默认 M1 ``validate_graph``，全部 V-* 检查）。
   不过 → 把 ValidationError 列表与自检问题回灌 LLM 重试（≤ max_retries），
   仍不过 → 抛 :class:`CompileError`（**不绕过校验**）。
4. **编译器自检**（SPEC §6，验收口径）：
   - 领域层 ≥3 个正交方向（数量检查 + 每个 rationale 非空）；
   - 每个假说 mechanism 与 prior 非空；
   - card_decisions 覆盖卡库全部卡 id（adopt/reject 均须有非空 rationale）。
   自检失败与校验失败同路处理（回灌重试）。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import yaml

from ..validator import ValidationError, validate_graph
from . import prompts
from .llm_client import LLMClient

__all__ = [
    "CompileError",
    "CompileResult",
    "compile_graph",
    "load_card_ids",
    "self_check_graph",
    "self_check_card_decisions",
]

#: 知识卡 id 提取：Markdown 三级标题「### A1 单原子变更原则 [动作]」
_CARD_ID_RE = re.compile(r"^###\s+([A-Z]\d+)\b", re.MULTILINE)

#: 代码围栏（```yaml / ```json / 无语言标注）
_FENCE_RE = re.compile(r"```(?:[a-zA-Z0-9_-]+)?\s*\n(.*?)```", re.DOTALL)

_MIN_DOMAINS = 3


class CompileError(RuntimeError):
    """编译失败：重试耗尽仍未通过解析 / 自检 / 确定性校验门。"""

    def __init__(self, stage: str, attempts: int, problems: Sequence[str]) -> None:
        self.stage = stage
        self.attempts = attempts
        self.problems = list(problems)
        detail = "\n".join(f"  - {p}" for p in self.problems[:20])
        super().__init__(
            f"编译失败（阶段 {stage}，已重试 {attempts} 次，不绕过校验门）:\n{detail}"
        )


@dataclass(frozen=True)
class CompileResult:
    """编译产物（SPEC §6）。

    ``graph_yaml`` 为解析成功的干净 YAML 文本（保留 LLM 原始排版，已去围栏/杂讯，
    保证 ``yaml.safe_load(graph_yaml) == graph``）；``attempts`` = 成图步 LLM
    调用次数（含成功那次）；``select_attempts`` = 选卡步调用次数。
    """

    graph: dict[str, Any]
    graph_yaml: str
    card_decisions: list[dict[str, Any]]
    attempts: int
    select_attempts: int = field(default=1)


# --------------------------------------------------------------------------
# 输入加载
# --------------------------------------------------------------------------


def load_card_ids(cards_path: str | Path) -> list[str]:
    """从知识卡库 Markdown 提取全部卡 id（按出现顺序）。"""
    text = Path(cards_path).read_text(encoding="utf-8")
    ids = _CARD_ID_RE.findall(text)
    if not ids:
        raise CompileError("选卡", 0, [f"卡库中未解析到任何卡 id: {cards_path}"])
    return ids


# --------------------------------------------------------------------------
# LLM 输出鲁棒解析（容忍围栏与首尾杂讯）
# --------------------------------------------------------------------------


def _candidates(text: str, start_pattern: re.Pattern[str]) -> list[str]:
    """解析候选串：优先第一个代码围栏内容，其次全文，再其次从首个匹配行截断。"""
    out: list[str] = []
    m = _FENCE_RE.search(text)
    if m:
        out.append(m.group(1).strip())
    stripped = text.strip()
    out.append(stripped)
    m2 = start_pattern.search(stripped)
    if m2 and m2.start() > 0:
        out.append(stripped[m2.start():].strip())
    return out


def _parse_json_payload(text: str) -> Any:
    """鲁棒 JSON 解析：围栏 → 全文 → 首个 [ / { 起切片。"""
    errs: list[str] = []
    cands = _candidates(text, re.compile(r"[\[{]"))
    # 额外候选：首个 [ 到末个 ]（论证表是数组）
    lo, hi = text.find("["), text.rfind("]")
    if 0 <= lo < hi:
        cands.append(text[lo:hi + 1])
    for cand in cands:
        try:
            return json.loads(cand)
        except json.JSONDecodeError as exc:
            errs.append(str(exc))
    raise ValueError(f"JSON 解析失败: {errs[-1] if errs else '空输出'}")


def _parse_yaml_payload(text: str) -> tuple[dict[str, Any], str]:
    """鲁棒 YAML 解析：围栏 → 全文 → 从首个 version: 行截断。要求顶层为映射。

    返回 ``(doc, cleaned_yaml)``：cleaned_yaml 为解析成功的候选串
    （保留 LLM 原始排版，去除围栏与首尾杂讯），保证 ``safe_load`` 可得 doc。
    """
    errs: list[str] = []
    for cand in _candidates(text, re.compile(r"(?m)^version\s*:")):
        try:
            doc = yaml.safe_load(cand)
        except yaml.YAMLError as exc:
            errs.append(str(exc))
            continue
        if isinstance(doc, dict):
            return doc, cand
        errs.append(f"顶层不是映射（{type(doc).__name__}）")
    raise ValueError(f"YAML 解析失败: {errs[-1] if errs else '空输出'}")


# --------------------------------------------------------------------------
# 编译器自检（SPEC §6 验收口径；与 M1 校验器互补，不重复 V-* 检查）
# --------------------------------------------------------------------------


def _non_empty(x: Any) -> bool:
    return isinstance(x, str) and bool(x.strip())


def self_check_graph(graph: Mapping[str, Any]) -> list[str]:
    """图级自检：领域 ≥3 且 rationale 非空；每假说 mechanism/prior 非空。"""
    problems: list[str] = []
    nodes = [n for n in (graph.get("nodes") or []) if isinstance(n, Mapping)]
    domains = [n for n in nodes if n.get("layer") == "domain"]
    if len(domains) < _MIN_DOMAINS:
        problems.append(
            f"领域层仅 {len(domains)} 个方向（要求 ≥{_MIN_DOMAINS} 个正交方向）"
        )
    for d in domains:
        if not _non_empty(d.get("rationale")):
            problems.append(f"domain {d.get('id')!r} 的 rationale 为空（须有选域论证）")
    for h in (n for n in nodes if n.get("layer") == "hypothesis"):
        if not _non_empty(h.get("mechanism")):
            problems.append(f"hypothesis {h.get('id')!r} 的 mechanism 为空（须有机制解释）")
        prior = h.get("prior")
        if prior is None or (isinstance(prior, str) and not prior.strip()):
            problems.append(f"hypothesis {h.get('id')!r} 的 prior 为空（须标注先验）")
    return problems


def self_check_card_decisions(
    decisions: Any, card_ids: Iterable[str]
) -> list[str]:
    """论证表自检：结构合法、覆盖卡库全部卡 id、adopt/reject 均有非空 rationale。"""
    problems: list[str] = []
    if not isinstance(decisions, list):
        return [f"选卡论证表必须是 JSON 数组，得到 {type(decisions).__name__}"]
    seen: dict[str, Mapping[str, Any]] = {}
    for i, item in enumerate(decisions):
        if not isinstance(item, Mapping):
            problems.append(f"论证表第 {i} 项不是对象")
            continue
        cid = item.get("card_id")
        if not _non_empty(cid):
            problems.append(f"论证表第 {i} 项缺少 card_id")
            continue
        cid = str(cid).strip()
        if cid in seen:
            problems.append(f"card_id {cid!r} 重复出现")
        seen[cid] = item
        decision = item.get("decision")
        if decision not in ("adopt", "reject"):
            problems.append(f"卡 {cid} 的 decision 必须是 adopt|reject，得到 {decision!r}")
        if not _non_empty(item.get("rationale")):
            problems.append(f"卡 {cid} 缺少非空 rationale（采用/拒绝均须有论证）")
    expected = set(card_ids)
    missing = expected - set(seen)
    extra = set(seen) - expected
    if missing:
        problems.append(f"论证表未覆盖卡库全部卡 id，缺失: {sorted(missing)}")
    if extra:
        problems.append(f"论证表包含卡库之外的 id: {sorted(extra)}")
    return problems


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------


def _normalized_decisions(raw: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in raw:
        out.append(
            {
                "card_id": str(item.get("card_id", "")).strip(),
                "decision": str(item.get("decision", "")).strip(),
                "rationale": str(item.get("rationale", "")).strip(),
            }
        )
    return out


def compile_graph(
    cards_path: str | Path,
    brief_path: str | Path,
    client: LLMClient,
    validator: Callable[[Mapping[str, Any]], list[ValidationError]] = validate_graph,
    max_retries: int = 3,
) -> CompileResult:
    """编译编排图（SPEC §6 签名）。

    :param cards_path: 知识卡库 Markdown；:param brief_path: 任务书+地形 YAML；
    :param client: 可替换 LLM 客户端；:param validator: 确定性校验门
    （默认 M1 :func:`validate_graph`）；:param max_retries: 每步最大尝试次数。
    :raises CompileError: 重试耗尽仍未通过（不绕过校验）。
    """
    if max_retries < 1:
        raise ValueError("max_retries 必须 >= 1")
    cards_text = Path(cards_path).read_text(encoding="utf-8")
    brief_text = Path(brief_path).read_text(encoding="utf-8")
    card_ids = load_card_ids(cards_path)

    # ---- S4a 选卡：解析 + 自检，失败回灌重试 ------------------------------
    decisions: list[dict[str, Any]] | None = None
    feedback: list[str] = []
    select_attempts = 0
    for attempt in range(1, max_retries + 1):
        select_attempts = attempt
        raw = client.complete(
            prompts.CARD_SELECTION_SYSTEM,
            prompts.card_selection_user(cards_text, brief_text, errors=feedback),
        )
        try:
            payload = _parse_json_payload(raw)
        except ValueError as exc:
            feedback = [f"选卡输出解析失败: {exc}"]
            continue
        problems = self_check_card_decisions(payload, card_ids)
        if not problems:
            decisions = _normalized_decisions(payload)
            break
        feedback = problems
    if decisions is None:
        raise CompileError("选卡", select_attempts, feedback)

    # ---- S4b 成图 + 校验门：不过则回灌 ValidationError 重试 ----------------
    feedback = []
    for attempt in range(1, max_retries + 1):
        graph_yaml = client.complete(
            prompts.GRAPH_SYSTEM,
            prompts.graph_user(cards_text, brief_text, decisions, errors=feedback),
        )
        try:
            graph, clean_yaml = _parse_yaml_payload(graph_yaml)
        except ValueError as exc:
            feedback = [f"成图输出解析失败: {exc}"]
            continue
        problems = [str(e) for e in validator(graph)]
        problems += self_check_graph(graph)
        if not problems:
            return CompileResult(
                graph=graph,
                graph_yaml=clean_yaml,
                card_decisions=decisions,
                attempts=attempt,
                select_attempts=select_attempts,
            )
        feedback = problems
    raise CompileError("成图", max_retries, feedback)
