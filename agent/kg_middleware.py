"""T2 知识图谱 Middleware：meta-plan 视图注入 + kg_* 三工具 + E9 动态重排。

语义来源：《01_设计书_T2知识图谱Middleware.md》§3 D1~D6、§4 接口契约。

- §4.3：`wrap_model_call` 每次把当前图状态渲染为「meta-plan 视图」追加到
  `ModelRequest.system_message`（视图定位 = 信息供给而非路线推荐——用户拍板
  D2/D5 定位修正：禁用"推荐/建议执行"措辞，优先级数字以"参考信息"呈现，
  refuted/blocked 的分支切换与冻结以"图结构提示"呈现）；整体 ≤2000 字符。
- §4.2：middleware 注册 `kg_report_evidence` / `kg_next` / `kg_decision` 三工具
  （签名逐字按契约）；非法 node_id（不在图内）→ 返回错误文本，不落盘；
  `kg_decision` 的 reason 强制非空，空则拒绝。
- §3-D5（E9）：priority(H) = posterior × parse_delta_mid(expected.delta)
  / (recon_steps + run_minutes)；后验更新复用 M3 `ledger.update_posterior`
  （单一事实源，禁止复刻公式）；refuted→alternatives 组内兄弟 boost；
  blocked→requires 传递下游冻结；uncertain→posterior 不变、排序 ×uncertain_dampen；
  同分 id 升序。
- §3-D3 / §4.4：state.json 与 hypotheses.jsonl 写入 `/memories/kg/` 前缀
  （CompositeBackend 的 ("filesystem",) 命名空间），仿主系统既有 store 访问方式
  （`_upload_single_skill_file` 的 create_file_data 写法 / `_export_plan_log` 的读法）。

工程默认值（设计书语义空白，列入交付说明「新发现」）：
1. 节点初始状态词表 `"pending"`（视图渲染"未验"）；已终局 = confirmed|refuted。
2. 后验挂在每个节点上：hypothesis 初值 = prior（契约）；action 无 prior 字段，
   工程默认值 0.5——账本 record_schema 要求 posterior ∈ [0,1]，action 回写时以其
   自身 shadow 后验记账（更新同样只走 `ledger.update_posterior`，公式不复刻）。
3. "验收进展"口径：goal.acceptance 断言的 metric 与某 confirmed 假说
   expected.metric 相同 → 计为"已有 confirmed 证据"；断言是否真正满足需实测
   指标，middleware 无法自判，视图如实标注。
4. kg_decision"分支语义为切换假说"的判定：扫描 branch 文本中出现的图内假说 id，
   命中即对该假说触发 boost（与 refuted 兄弟 boost 同一机制，决策来源落 state）。
5. blocked 冻结范围：被 blocked 节点本身 + （假说被 blocked 时）其所属 action
   + requires 边传递下游（from→to 方向，即 to requires from）；冻结为派生集合，
   每次证据回写后按当前 blocked 节点全集重算（后续 confirmed 可解冻）。
6. boost 操作化：boosted 假说排在未决假说最前（组内仍按 priority 降序、id 升序）。
7. 视图超预算截断：按 E9 排序保留最前 k 条未决假说，已终局假说只保留计数；
   仍超长则尾部硬截断（预算默认 2000 字符，`view_char_budget` 可覆盖）。
8. 成本分母为 0（recon_steps + run_minutes == 0）时按 1.0 计（防零除，合法图
   成本字段必填但允许 0）。
9. M3 `Ledger` 是文件路径实现：本模块复用 `Ledger.validate_record` 做账本记录
   校验、复用 `update_posterior` 做后验更新（单一事实源）；存储改走 backend
   store——store 无追加原语，`hypotheses.jsonl` 采用"读旧内容 + 拼接新行 + 整体
   写回"，append-only 语义由"只增行、不改旧行"保证（薄适配层）。
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any, Callable, Literal

from langchain.agents.middleware.types import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import SystemMessage
from langchain_core.tools import StructuredTool

from deepagents.backends.utils import create_file_data

from knowledge_pipeline.ledger import STATUSES, Ledger, update_posterior
from knowledge_pipeline.simulator.scheduler import parse_delta_mid

# ---- 契约常量（§4.3 / §4.4） ------------------------------------------------

#: store 命名空间与键（/memories/kg/ 经 CompositeBackend 路由到 ("filesystem",)）
KG_NAMESPACE = ("filesystem",)
KG_STATE_KEY = "/kg/state.json"
KG_LEDGER_KEY = "/kg/hypotheses.jsonl"

STATE_VERSION = "1.0"
INITIAL_STATE = "pending"
TERMINAL_STATES = ("confirmed", "refuted")
DEFAULT_VIEW_CHAR_BUDGET = 2000
DEFAULT_UNCERTAIN_DAMPEN = 0.5
#: 非 hypothesis 节点的 shadow 后验初值（工程默认值 2）
DEFAULT_ACTION_PRIOR = 0.5

_STATE_LABELS = {
    "pending": "未验",
    "confirmed": "confirmed",
    "refuted": "refuted",
    "uncertain": "曾判 uncertain",
    "blocked": "blocked",
}

_DISCIPLINE_LINE = "纪律：验证结果用 kg_report_evidence 回写；决策点用 kg_decision 落盘（理由必填）。"
_SORT_RULE_LINE = (
    "排序口径（参考信息）：posterior × 预期收益中值 / (recon步数+run分钟)；"
    "uncertain 降权；备选组兄弟已 refuted 的分支置前（图结构提示）；同分按 id 升序。"
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _short(text: Any, limit: int = 40) -> str:
    """单行截断（视图渲染用）。"""
    s = str(text or "").replace("\n", " ").strip()
    return s if len(s) <= limit else s[: limit - 1] + "…"


class KnowledgeGraphMiddleware(AgentMiddleware):
    """知识图谱 Middleware（仅挂 Planner，设计书 §3-D1）。

    参数：
    - graph：已过 M1 校验的编排图 dict（启动接线保证，见 agent.py）；
    - store：主系统共享 store（CompositeBackend `/memories/` 路由的目标）；
    - graph_path：图来源路径（落 state.json，便于审计）；
    - uncertain_dampen：uncertain 假说的排序降权系数（§4.1，默认 0.5）；
    - view_char_budget：视图渲染预算（§4.3，默认 2000 字符）。
    """

    def __init__(
        self,
        graph: dict,
        store: Any,
        graph_path: str = "",
        uncertain_dampen: float = DEFAULT_UNCERTAIN_DAMPEN,
        view_char_budget: int = DEFAULT_VIEW_CHAR_BUDGET,
    ) -> None:
        self.graph = graph
        self.store = store
        self.graph_path = graph_path
        self.uncertain_dampen = float(uncertain_dampen)
        self.view_char_budget = int(view_char_budget)

        # ---- 图静态索引 ----
        self._nodes: dict[str, dict] = {
            n["id"]: n for n in graph.get("nodes", []) if isinstance(n, dict) and "id" in n
        }
        self._decisions: dict[str, dict] = {
            d["id"]: d for d in graph.get("decisions", []) if isinstance(d, dict) and "id" in d
        }
        self._hypotheses: dict[str, dict] = {
            nid: n for nid, n in self._nodes.items() if n.get("layer") == "hypothesis"
        }
        self._actions_of: dict[str, list[str]] = {}
        for nid, n in self._nodes.items():
            if n.get("layer") == "action":
                self._actions_of.setdefault(n.get("belongs_to"), []).append(nid)
        self._requires_downstream: dict[str, set[str]] = {}
        self._alt_siblings: dict[str, set[str]] = {}
        self._no_parallel: list[tuple[str, str, str]] = []
        for e in graph.get("edges", []) or []:
            if not isinstance(e, dict):
                continue
            frm, to, etype = e.get("from"), e.get("to"), e.get("type")
            if etype == "requires":
                self._requires_downstream.setdefault(frm, set()).add(to)
            elif etype == "alternatives":
                self._alt_siblings.setdefault(frm, set()).add(to)
                self._alt_siblings.setdefault(to, set()).add(frm)
            elif etype == "no_parallel":
                self._no_parallel.append((frm, to, str(e.get("reason") or "")))

        # ---- 运行态（§4.4 state.json 的内存镜像） ----
        self._node_state: dict[str, dict] = {nid: {"state": INITIAL_STATE} for nid in self._nodes}
        self._posterior: dict[str, float] = {
            nid: float(n.get("prior", DEFAULT_ACTION_PRIOR)) for nid, n in self._nodes.items()
        }
        self._decisions_made: dict[str, dict] = {}
        self._gain_history: dict[str, list[float]] = {}
        self._refresh_count = 0
        # ---- 派生/辅助状态 ----
        self._evidence_log: dict[str, str] = {}
        self._frozen: set[str] = set()
        self._frozen_cause: dict[str, str] = {}
        self._decision_boosted: set[str] = set()

        self.tools = [
            StructuredTool.from_function(
                func=self.kg_report_evidence,
                name="kg_report_evidence",
                description=(
                    "回写某假说/动作的验证结果（知识图谱账本）。状态机迁移 + 账本 append + "
                    "后验更新 + E9 重排；返回重排后的假说全景。node_id 必须在图内，否则拒绝落盘。"
                ),
            ),
            StructuredTool.from_function(
                func=self.kg_next,
                name="kg_next",
                description=(
                    "查询知识图谱当前假说全景与排序理由（无副作用，信息供给："
                    "E9 排序、备选组切换与冻结的图结构提示）。"
                ),
            ),
            StructuredTool.from_function(
                func=self.kg_decision,
                name="kg_decision",
                description=(
                    "决策点判定落盘（分支 + 理由）。reason 强制非空，空则拒绝；"
                    "分支文本命中图内假说 id 时触发对应分支置前。"
                ),
            ),
        ]
        self._persist_state()

    # ------------------------------------------------------------------
    # §4.3 wrap_model_call：meta-plan 视图注入
    # ------------------------------------------------------------------

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """每次 Planner 调模型前：刷新计数 → 现渲染视图 → 追加到 system_message。

        视图每次现渲染现注入（不依赖历史消息存活，免疫摘要丢失，设计书 §9 开放问题 2）。
        """
        self._refresh_count += 1
        view = self._render_view()
        existing = request.system_message
        if existing is not None:
            new_content = list(existing.content_blocks) + [{"type": "text", "text": "\n\n" + view}]
        else:
            new_content = view
        self._persist_state()
        return handler(request.override(system_message=SystemMessage(content=new_content)))

    # ------------------------------------------------------------------
    # §4.2 三个工具（签名逐字按契约）
    # ------------------------------------------------------------------

    def kg_report_evidence(
        self,
        node_id: str,
        status: Literal["confirmed", "refuted", "uncertain", "blocked"],
        evidence: str,
        measured_gain: float | None = None,
    ) -> str:
        """回写某假说/动作的验证结果；返回重排后的假说全景文本。"""
        if node_id not in self._nodes:
            return (
                f"❌ kg_report_evidence 被拒绝：node_id={node_id!r} 不在图内"
                f"（图内节点：{sorted(self._nodes)}）。未落盘。"
            )
        if status not in STATUSES:
            return f"❌ kg_report_evidence 被拒绝：status={status!r} 非法（词表：{STATUSES}）。未落盘。"

        # 1) 状态机迁移
        self._node_state[node_id]["state"] = status
        # 2) 后验更新（单一事实源：M3 ledger.update_posterior）
        self._posterior[node_id] = update_posterior(self._posterior[node_id], status)
        # 3) gain 历史 / 证据
        if measured_gain is not None:
            self._gain_history.setdefault(node_id, []).append(float(measured_gain))
        self._evidence_log[node_id] = str(evidence)
        # 4) blocked → requires 传递下游冻结（派生集合重算）
        self._recompute_frozen()
        # 5) 账本 append（M3 记录 schema，校验复用 Ledger.validate_record）
        record = {
            "node_id": node_id,
            "status": status,
            "evidence": str(evidence),
            "metrics": {"gain": float(measured_gain)} if measured_gain is not None else {},
            "posterior": self._posterior[node_id],
            "notes": "uncertain：边际提升未达阈，不计入战果" if status == "uncertain" else "",
            "ts": _utc_now_iso(),
        }
        Ledger.validate_record(record)  # 非法记录不落盘（防污染账本）
        self._append_ledger_line(json.dumps(record, ensure_ascii=False))
        # 6) 状态落盘 + E9 重排后返回全景
        self._persist_state()
        header = (
            f"✅ 已回写：{node_id} → {status}（posterior={self._posterior[node_id]:.2f}）。"
            "以下为 E9 重排后的假说全景（信息供给，行动判断归你）："
        )
        return header + "\n" + self._render_panorama(with_reasons=True)

    def kg_next(self) -> str:
        """查询当前假说全景与排序理由（无副作用：不迁移状态、不落盘、不计刷新）。"""
        return "知识图谱当前假说全景与排序理由（信息供给，行动判断归你）：\n" + self._render_panorama(
            with_reasons=True
        )

    def kg_decision(self, decision_id: str, branch: str, reason: str) -> str:
        """决策点判定落盘；reason 强制非空，空则拒绝（决策落盘纪律）。"""
        if decision_id not in self._decisions:
            return (
                f"❌ kg_decision 被拒绝：decision_id={decision_id!r} 不在图内"
                f"（图内决策点：{sorted(self._decisions)}）。未落盘。"
            )
        if not isinstance(reason, str) or not reason.strip():
            return "❌ kg_decision 被拒绝：reason 必须非空（决策落盘纪律：放过检查点必须留下理由）。未落盘。"
        self._decisions_made[decision_id] = {
            "branch": str(branch),
            "reason": reason,
            "ts": _utc_now_iso(),
        }
        # 分支语义为切换假说 → 对应假说 boost（工程默认值 4：按 branch 文本命中假说 id 判定）
        switched = sorted(
            hid for hid in self._hypotheses if re.search(rf"(?<![A-Za-z0-9]){re.escape(hid)}(?![0-9])", str(branch))
        )
        self._decision_boosted.update(switched)
        self._persist_state()
        msg = f"✅ 决策已落盘：{decision_id} → 分支「{_short(branch, 60)}」（理由已记录）。"
        if switched:
            msg += f" 图结构提示：分支涉及假说 {switched}，已置前（备选/切换信息，供你判断参考）。"
        return msg

    # ------------------------------------------------------------------
    # E9（§3-D5）：优先级、boost、冻结
    # ------------------------------------------------------------------

    def _priority(self, nid: str) -> float:
        """priority(H) = posterior × parse_delta_mid(expected.delta) / (recon_steps + run_minutes)。"""
        node = self._hypotheses[nid]
        posterior = self._posterior[nid]
        delta_mid = parse_delta_mid((node.get("expected") or {}).get("delta"))
        cost = node.get("cost") or {}
        denom = (cost.get("recon_steps") or 0) + (cost.get("run_minutes") or 0)
        if denom <= 0:
            denom = 1.0  # 工程默认值 8：防零除
        priority = posterior * delta_mid / denom
        if self._node_state[nid]["state"] == "uncertain":
            priority *= self.uncertain_dampen
        return priority

    def _boosted_set(self) -> set[str]:
        """boost 全集 = refuted 假说的 alternatives 组内未终局兄弟 ∪ 决策分支切换到的假说。"""
        boosted = set(self._decision_boosted)
        for nid, node in self._hypotheses.items():
            if self._node_state[nid]["state"] != "refuted":
                continue
            for sib in self._alt_siblings.get(nid, ()):
                if sib in self._hypotheses and self._node_state[sib]["state"] not in TERMINAL_STATES:
                    boosted.add(sib)
        return boosted

    def _recompute_frozen(self) -> None:
        """blocked → 自身 + 所属 action + requires 传递下游冻结（工程默认值 5）。"""
        frozen: set[str] = set()
        cause: dict[str, str] = {}
        for nid, st in self._node_state.items():
            if st["state"] != "blocked":
                continue
            seeds = {nid} | set(self._actions_of.get(nid, []))
            stack = list(seeds)
            while stack:
                cur = stack.pop()
                if cur in frozen:
                    continue
                frozen.add(cur)
                cause.setdefault(cur, nid)
                stack.extend(self._requires_downstream.get(cur, ()))
        self._frozen = frozen
        self._frozen_cause = cause

    def _rank_unresolved(self) -> list[tuple[str, float]]:
        """未决且未冻结假说的 E9 排序：boosted 在前 → priority 降序 → id 升序。"""
        boosted = self._boosted_set()
        rows = []
        for nid in self._hypotheses:
            state = self._node_state[nid]["state"]
            if state in TERMINAL_STATES or state == "blocked" or nid in self._frozen:
                continue
            rows.append((nid, self._priority(nid)))
        rows.sort(key=lambda r: (0 if r[0] in boosted else 1, -r[1], r[0]))
        return rows

    # ------------------------------------------------------------------
    # §4.3 视图渲染（四类内容分工 + 纪律行；≤2000 字符预算）
    # ------------------------------------------------------------------

    def _render_view(self) -> str:
        goal = self.graph.get("goal") or {}
        acceptance = goal.get("acceptance") or []
        covered = self._acceptance_covered_count(acceptance)

        header = [
            f"[知识图谱 · meta-plan 视图]（第 {self._refresh_count} 次刷新）",
            f"目标：{goal.get('statement', '')}",
            f"验收进展：{covered}/{len(acceptance)} 条验收断言已有 confirmed 证据"
            "（断言是否真正达标以实测指标为准）",
            "",
        ]
        panorama = self._render_panorama(with_reasons=False)
        facts = self._render_confirmed_facts()
        conflicts = self._render_conflicts()
        pending = self._render_pending_decisions()
        full = "\n".join(header + [panorama, facts, conflicts, pending, _DISCIPLINE_LINE])
        if len(full) <= self.view_char_budget:
            return full
        # 超预算：按 E9 排序截断假说列表（已终局的只保留计数，工程默认值 7）
        ranked = self._rank_unresolved()
        for keep in range(len(ranked), -1, -1):
            panorama = self._render_panorama(with_reasons=False, keep=keep)
            full = "\n".join(header + [panorama, facts, conflicts, pending, _DISCIPLINE_LINE])
            if len(full) <= self.view_char_budget:
                return full
        # 仍超长（如目标/冲突文本本身超长）：尾部硬截断
        return full[: self.view_char_budget - 1] + "…"

    def _render_panorama(self, with_reasons: bool, keep: int | None = None) -> str:
        """假说全景（按 E9 排序；boost/uncertain/终局计数标注）。"""
        ranked = self._rank_unresolved()
        total_unresolved = len(ranked)
        if keep is not None:
            ranked = ranked[:keep]
        boosted = self._boosted_set()
        lines = ["假说全景（按后验排列，仅供你判断参考）："]
        if not ranked:
            lines.append(" · （无未决假说）")
        for nid, prio in ranked:
            node = self._hypotheses[nid]
            state = self._node_state[nid]["state"]
            label = _STATE_LABELS.get(state, state)
            head = f" · {nid} {_short(node.get('title'), 30)} [posterior {self._posterior[nid]:.2f}，{label}]"
            if with_reasons:
                head += f" priority参考 {prio:.4f}"
            if nid in boosted:
                refuted_sibs = sorted(
                    s for s in self._alt_siblings.get(nid, ())
                    if self._node_state.get(s, {}).get("state") == "refuted"
                )
                if refuted_sibs:
                    head += (
                        f"\n   ⚠ 备选组内兄弟假说 {'、'.join(refuted_sibs)} 已 refuted"
                        "——图结构提示：该备选方向仅剩此分支"
                    )
                else:
                    head += "\n   ⚠ 决策分支切换涉及此假说——图结构提示：已置前"
            lines.append(head)
            mech = _short(node.get("mechanism"), 60)
            if mech:
                lines.append(f"   机制：{mech}")
            actions = self._actions_of.get(nid, [])
            if actions:
                briefs = []
                for aid in sorted(actions):
                    a = self._nodes[aid]
                    mark = "（冻结中）" if aid in self._frozen else ""
                    briefs.append(f"{aid}（{a.get('type', '?')}，{_short(a.get('do'), 30)}）{mark}")
                lines.append(f"   关联动作：{'、'.join(briefs)}")
        # 终局/冻结计数（截断时只保留计数）
        n_confirmed = sum(1 for nid in self._hypotheses if self._node_state[nid]["state"] == "confirmed")
        n_refuted = sum(1 for nid in self._hypotheses if self._node_state[nid]["state"] == "refuted")
        n_blocked = sum(
            1 for nid in self._hypotheses
            if self._node_state[nid]["state"] == "blocked" or nid in self._frozen
        )
        counts = []
        if n_confirmed or n_refuted:
            counts.append(f"已终局 {n_confirmed + n_refuted} 条（confirmed {n_confirmed} / refuted {n_refuted}）")
        if n_blocked:
            counts.append(f"冻结 {n_blocked} 条")
        omitted = total_unresolved - len(ranked)
        if omitted > 0:
            counts.append(f"另有 {omitted} 条未决假说因视图预算未列出")
        if counts:
            lines.append("（" + "；".join(counts) + "）")
        # 冻结的图结构提示（blocked 下游：Planner 自己算不出的信息，§3-D5 定位修正②）
        if self._frozen:
            frozen_ids = sorted(n for n in self._frozen if n in self._nodes)
            causes = sorted({self._frozen_cause.get(n, "?") for n in frozen_ids})
            lines.append(
                f"图结构提示：{'、'.join(frozen_ids)} 冻结中（上游 {'、'.join(causes)} blocked，已退出待办）"
            )
        if with_reasons:
            lines.append(_SORT_RULE_LINE)
        return "\n".join(lines)

    def _render_confirmed_facts(self) -> str:
        facts = []
        for nid in sorted(self._nodes):
            if self._node_state[nid]["state"] != "confirmed":
                continue
            node = self._nodes[nid]
            evidence = _short(self._evidence_log.get(nid, ""), 50)
            entry = f"{nid} {_short(node.get('title') or node.get('do'), 24)}"
            if evidence:
                entry += f"（evidence: {evidence}）"
            facts.append(entry)
        return "已确认事实：" + ("；".join(facts) if facts else "（暂无）")

    def _render_conflicts(self) -> str:
        parts = []
        for frm, to, reason in self._no_parallel:
            entry = f"{frm} 与 {to} 互斥（no_parallel"
            if reason:
                entry += f"：{_short(reason, 40)}"
            entry += "）"
            parts.append(entry)
        frozen_sorted = sorted(nid for nid in self._frozen if nid in self._nodes)
        if frozen_sorted:
            causes = sorted({self._frozen_cause.get(nid, "?") for nid in frozen_sorted})
            parts.append(f"{'、'.join(frozen_sorted)} 冻结中（上游 {'、'.join(causes)} blocked）")
        return "冲突与约束：" + ("；".join(parts) if parts else "（无）")

    def _render_pending_decisions(self) -> str:
        pending = []
        for did in sorted(self._decisions):
            if did in self._decisions_made:
                continue
            d = self._decisions[did]
            pending.append(f"{did}（在 {d.get('after', '?')} 完成后，读 {d.get('read', '?')} 判定分支）")
        return "待决决策点：" + ("；".join(pending) if pending else "（无）")

    def _acceptance_covered_count(self, acceptance: list) -> int:
        """验收进展口径（工程默认值 3）：断言 metric 被某 confirmed 假说 expected.metric 覆盖。"""
        confirmed_metrics = set()
        for nid, node in self._hypotheses.items():
            if self._node_state[nid]["state"] == "confirmed":
                metric = (node.get("expected") or {}).get("metric")
                if metric:
                    confirmed_metrics.add(str(metric))
        covered = 0
        for assertion in acceptance:
            text = str(assertion)
            if any(m and m in text for m in confirmed_metrics):
                covered += 1
        return covered

    # ------------------------------------------------------------------
    # §3-D3 / §4.4 存储（/memories/kg/，走 backend store）
    # ------------------------------------------------------------------

    def _state_doc(self) -> dict:
        nodes_doc: dict[str, dict] = {}
        for nid, st in self._node_state.items():
            entry: dict[str, Any] = {"state": st["state"]}
            if nid in self._hypotheses:
                entry["posterior"] = self._posterior[nid]
            nodes_doc[nid] = entry
        return {
            "version": STATE_VERSION,
            "graph_path": self.graph_path,
            "nodes": nodes_doc,
            "decisions": self._decisions_made,
            "gain_history": self._gain_history,
            "refresh_count": self._refresh_count,
        }

    def _persist_state(self) -> None:
        self._store_put_text(KG_STATE_KEY, json.dumps(self._state_doc(), ensure_ascii=False, indent=2))

    def _append_ledger_line(self, line: str) -> None:
        """账本 append（工程默认值 9：store 无追加原语，读旧 + 拼新行 + 写回）。"""
        old = self._store_read_text(KG_LEDGER_KEY)
        new = (old + "\n" + line) if old else line
        self._store_put_text(KG_LEDGER_KEY, new)

    def _store_put_text(self, key: str, text: str) -> None:
        self.store.put(KG_NAMESPACE, key, create_file_data(text))

    def _store_read_text(self, key: str) -> str:
        item = self.store.get(KG_NAMESPACE, key)
        if not item:
            return ""
        raw = item.value
        content = raw.get("content", "") if isinstance(raw, dict) else getattr(raw, "content", "")
        if isinstance(content, list):
            return "\n".join(str(line) for line in content)
        return str(content or "")
