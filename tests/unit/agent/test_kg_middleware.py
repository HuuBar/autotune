"""W-B 知识图谱 Middleware 单测（mock LLM，不依赖真实端点）。

覆盖（对应工作包 W-B 验收 a~f）：
a) 工具三件套签名与行为（合法/非法 node_id、reason 空拒绝）；
b) 状态机迁移 + 后验锚点（confirmed 0.5→0.95，经由 ledger.update_posterior）；
c) E9：refuted→兄弟 boost、blocked→下游冻结、uncertain→降权、同分 id 升序；
d) 视图渲染：含四类内容、≤2000 字符截断、无"推荐/建议执行"类指令措辞；
e) 启动接线：合法图挂载 / 非法图 abort / 非法图 warn_disable 等价关闭 /
   enabled 缺省时 _get_middleware 列表与现状相同；
f) 导出：finally 阶段导出函数产出 kg/ 两文件且 schema 合法。
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest
from langchain_core.messages import SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from agent.agent import AutoTuneAgent
from agent.factory import PlannerAgentBuilder
from agent.kg_middleware import (
    KG_LEDGER_KEY,
    KG_NAMESPACE,
    KG_STATE_KEY,
    KnowledgeGraphMiddleware,
)
from deepagents.backends.utils import create_file_data
from deepagents.middleware._tool_exclusion import _ToolExclusionMiddleware
from knowledge_pipeline import load_graph
from knowledge_pipeline.ledger import Ledger, update_posterior
from langchain.agents.middleware import ModelRetryMiddleware, SummarizationMiddleware
from utils.config import global_config

FIXTURES = Path(__file__).resolve().parents[3] / "knowledge_pipeline" / "fixtures"
GOLDEN = FIXTURES / "tbox_golden.yaml"

BANNED_WORDS = ("推荐", "建议")


# --------------------------------------------------------------------------
# 测试夹具
# --------------------------------------------------------------------------


class _FakeRequest:
    """最小 ModelRequest 替身：system_message + override（langchain 1.2.15 契约）。"""

    def __init__(self, system_message=None):
        self.system_message = system_message

    def override(self, **overrides):
        new = _FakeRequest(self.system_message)
        for key, value in overrides.items():
            setattr(new, key, value)
        return new


def _hyp(nid, prior, delta, recon=1, run=5):
    return {
        "id": nid,
        "layer": "hypothesis",
        "domain": "D1",
        "title": f"假说{nid}",
        "mechanism": f"{nid} 的机制说明",
        "prior": prior,
        "expected": {"metric": "overall.F1", "delta": delta, "confidence": "low"},
        "cost": {"recon_steps": recon, "run_minutes": run},
    }


def _act(aid, parent):
    return {
        "id": aid,
        "layer": "action",
        "belongs_to": parent,
        "type": "recon",
        "do": f"{aid} 的动作",
        "tool_hint": "read_file",
        "produces": f"p_{aid}",
        "verify": "v",
        "retry_budget": 1,
    }


def make_e9_graph():
    """E9 专用图：alternatives 组 (H3,H4)、requires 链 A1->A2（H1 所属）。"""
    return {
        "version": "1.0",
        "meta": {"task": "E9 测试", "created_by": "pytest", "source_cards": []},
        "goal": {"statement": "测试目标", "acceptance": ["overall.F1 >= 0.9"]},
        "nodes": [
            {"id": "D1", "layer": "domain", "title": "域一", "rationale": "r"},
            _hyp("H1", 0.9, "+0.05~0.15"),   # uncertain 降权对象
            _hyp("H2", 0.5, "+0.05~0.15"),
            _hyp("H3", 0.4, "+0.05~0.15"),   # H4 的 alternatives 兄弟
            _hyp("H4", 0.9, "+0.30"),        # 将被 refuted
            _hyp("H5", 0.5, "+0.10"),        # 与 H6 同分
            _hyp("H6", 0.5, "+0.10"),
            _act("A1", "H1"),
            _act("A2", "H1"),
        ],
        "edges": [
            {"from": "H3", "to": "H4", "type": "alternatives"},
            {"from": "A1", "to": "A2", "type": "requires"},
        ],
        "decisions": [
            {
                "id": "J1",
                "after": "A1",
                "read": "p_A1",
                "branches": [{"if": "x", "then": "切换 H3"}, {"else": "维持"}],
            }
        ],
        "writeback": {
            "ledger": "hypotheses.jsonl",
            "record_schema": "{node_id, status, evidence, metrics, posterior, notes, ts}",
            "after_domain_review": "r",
        },
    }


def make_mw(graph=None, **kwargs):
    store = InMemoryStore()
    mw = KnowledgeGraphMiddleware(
        graph=graph if graph is not None else load_graph(GOLDEN), store=store, **kwargs
    )
    return mw, store


def _state_doc(store):
    item = store.get(KG_NAMESPACE, KG_STATE_KEY)
    assert item, "state.json 未落盘"
    return json.loads(item.value["content"])


def _ledger_lines(store):
    item = store.get(KG_NAMESPACE, KG_LEDGER_KEY)
    if not item:
        return []
    return [l for l in item.value["content"].splitlines() if l.strip()]


def _render_view(mw, store):
    captured = {}

    def handler(request):
        captured["request"] = request
        return "ok"

    assert mw.wrap_model_call(_FakeRequest(SystemMessage(content="BASE")), handler) == "ok"
    sm = captured["request"].system_message
    assert sm.content_blocks[0]["text"] == "BASE"  # 原 system prompt 保留
    return sm.content_blocks[-1]["text"]


# --------------------------------------------------------------------------
# a) 工具三件套签名与行为
# --------------------------------------------------------------------------


def test_tools_registered_and_signatures():
    mw, _ = make_mw()
    assert [t.name for t in mw.tools] == ["kg_report_evidence", "kg_next", "kg_decision"]

    sig = inspect.signature(mw.kg_report_evidence)
    assert list(sig.parameters) == ["node_id", "status", "evidence", "measured_gain"]
    assert sig.parameters["measured_gain"].default is None
    assert list(inspect.signature(mw.kg_next).parameters) == []
    assert list(inspect.signature(mw.kg_decision).parameters) == ["decision_id", "branch", "reason"]

    schema_fields = set(mw.tools[0].args_schema.model_fields)
    assert schema_fields == {"node_id", "status", "evidence", "measured_gain"}


def test_report_evidence_valid_and_invalid_node_id():
    mw, store = make_mw()
    out = mw.kg_report_evidence("H1", "confirmed", "fit 调用缺参，见 train.py:88")
    assert "✅" in out and "H1" in out
    assert len(_ledger_lines(store)) == 1

    # 非法 node_id → 错误文本，不落盘（账本行数不变、state 无该节点）
    out = mw.kg_report_evidence("H99", "confirmed", "编造的节点")
    assert "❌" in out and "不在图内" in out
    assert len(_ledger_lines(store)) == 1
    assert "H99" not in _state_doc(store)["nodes"]


def test_report_evidence_tool_invoke_path():
    """经 StructuredTool.invoke（真实 LLM 工具调用路径）合法/非法入参。"""
    mw, store = make_mw()
    tool = {t.name: t for t in mw.tools}
    out = tool["kg_report_evidence"].invoke(
        {"node_id": "H2", "status": "confirmed", "evidence": "threshold_scan.json"}
    )
    assert "✅" in out
    out = tool["kg_report_evidence"].invoke({"node_id": "X1", "status": "confirmed", "evidence": "e"})
    assert "❌" in out and len(_ledger_lines(store)) == 1


def test_kg_decision_reason_required_and_unknown_id():
    mw, store = make_mw()
    assert "❌" in mw.kg_decision("J2", "先 A5", "")
    assert "❌" in mw.kg_decision("J2", "先 A5", "   ")
    assert "❌" in mw.kg_decision("J9", "先 A5", "正当理由是存在的")
    assert _state_doc(store)["decisions"] == {}

    out = mw.kg_decision("J2", "阈值+K折升级", "val 仅 200 条且 F1=1.0，疑过拟合")
    assert "✅" in out
    decision = _state_doc(store)["decisions"]["J2"]
    assert decision["branch"] == "阈值+K折升级"
    assert decision["reason"] == "val 仅 200 条且 F1=1.0，疑过拟合"
    assert decision["ts"]


def test_kg_decision_branch_switches_hypothesis_boost():
    """分支语义为切换假说 → 对应假说 boost（E9 图结构提示）。"""
    mw, _ = make_mw(graph=make_e9_graph())
    mw.kg_decision("J1", "切换 H3 分支", "A1 产物支持加权路线")
    ranked = [nid for nid, _ in mw._rank_unresolved()]
    assert ranked[0] == "H3"
    assert "决策分支切换涉及此假说" in mw.kg_next()


def test_kg_next_has_no_side_effects():
    mw, store = make_mw()
    before = store.get(KG_NAMESPACE, KG_STATE_KEY).value["content"]
    out1 = mw.kg_next()
    out2 = mw.kg_next()
    after = store.get(KG_NAMESPACE, KG_STATE_KEY).value["content"]
    assert before == after  # 无副作用：不落盘、不计刷新
    assert _state_doc(store)["refresh_count"] == 0
    assert "排序口径" in out1 and out1 == out2


# --------------------------------------------------------------------------
# b) 状态机迁移 + 后验锚点
# --------------------------------------------------------------------------


def test_posterior_anchor_confirmed_05_to_095():
    mw, store = make_mw()  # golden H1 prior = 0.5
    mw.kg_report_evidence("H1", "confirmed", "fit 调用缺参")
    assert _state_doc(store)["nodes"]["H1"] == {"state": "confirmed", "posterior": 0.95}
    # 锚点必须经由 ledger.update_posterior（单一事实源）
    assert _state_doc(store)["nodes"]["H1"]["posterior"] == update_posterior(0.5, "confirmed")
    record = json.loads(_ledger_lines(store)[0])
    Ledger.validate_record(record)  # 账本记录过 M3 校验
    assert record["posterior"] == 0.95 and record["node_id"] == "H1"


def test_state_machine_all_statuses():
    mw, store = make_mw()
    mw.kg_report_evidence("H1", "refuted", "e1")
    mw.kg_report_evidence("H2", "uncertain", "e2")
    mw.kg_report_evidence("H3", "blocked", "e3")
    nodes = _state_doc(store)["nodes"]
    assert nodes["H1"]["state"] == "refuted"
    assert nodes["H1"]["posterior"] == update_posterior(0.5, "refuted")  # golden H1 prior=0.5
    assert nodes["H2"]["state"] == "uncertain"
    assert nodes["H2"]["posterior"] == 0.7  # uncertain → posterior 不变
    assert nodes["H3"]["state"] == "blocked"
    assert nodes["H3"]["posterior"] == 0.6  # blocked → posterior 不变
    assert len(_ledger_lines(store)) == 3


def test_measured_gain_into_gain_history_and_metrics():
    mw, store = make_mw()
    mw.kg_report_evidence("A6", "confirmed", "run 完成", measured_gain=0.03)
    mw.kg_report_evidence("A6", "uncertain", "复跑边际", measured_gain=0.01)
    doc = _state_doc(store)
    assert doc["gain_history"]["A6"] == [0.03, 0.01]
    rec0, rec1 = (json.loads(l) for l in _ledger_lines(store))
    assert rec0["metrics"] == {"gain": 0.03}
    assert rec1["notes"]  # uncertain 记"不计入战果"
    # action 无 posterior 导出（§4.4 schema 样例：A2 只有 state）
    assert doc["nodes"]["A6"] == {"state": "uncertain"}


# --------------------------------------------------------------------------
# c) E9 动态重排
# --------------------------------------------------------------------------


def test_e9_refuted_triggers_sibling_boost():
    mw, _ = make_mw(graph=make_e9_graph())
    assert [nid for nid, _ in mw._rank_unresolved()][0] == "H4"  # 初始 H4 最高
    mw.kg_report_evidence("H4", "refuted", "回调 spw 实测 F1 下降")
    ranked = [nid for nid, _ in mw._rank_unresolved()]
    assert "H4" not in ranked  # 已终局不参与排序
    assert ranked[0] == "H3"  # 兄弟 boost 排到最前
    text = mw.kg_next()
    assert "兄弟假说 H4 已 refuted" in text and "图结构提示" in text


def test_e9_blocked_freezes_downstream():
    mw, _ = make_mw(graph=make_e9_graph())
    mw.kg_report_evidence("H1", "blocked", "沙箱缺 xgboost")
    assert {"H1", "A1", "A2"} <= mw._frozen  # 自身 + 所属 action + requires 下游
    ranked = [nid for nid, _ in mw._rank_unresolved()]
    assert "H1" not in ranked  # 冻结不出现在待办
    assert "冻结中" in mw.kg_next()


def test_e9_uncertain_dampens_ranking():
    mw, _ = make_mw(graph=make_e9_graph())
    before = [nid for nid, _ in mw._rank_unresolved()]
    assert before.index("H1") < before.index("H2")  # H1 prior 0.9 原本靠前
    mw.kg_report_evidence("H1", "uncertain", "边际提升复跑未达阈")
    after = [nid for nid, _ in mw._rank_unresolved()]
    assert after.index("H2") < after.index("H1")  # ×0.5 降权后落到 H2 之后
    assert mw._posterior["H1"] == 0.9  # posterior 不变
    # uncertain_dampen 可配置
    mw2, _ = make_mw(graph=make_e9_graph(), uncertain_dampen=0.0)
    mw2.kg_report_evidence("H1", "uncertain", "e")
    assert mw2._priority("H1") == 0.0


def test_e9_tie_break_id_ascending():
    mw, _ = make_mw(graph=make_e9_graph())
    h5 = mw._priority("H5")
    h6 = mw._priority("H6")
    assert h5 == h6 and h5 > 0
    ranked = [nid for nid, _ in mw._rank_unresolved()]
    assert ranked.index("H5") < ranked.index("H6")


def test_e9_priority_formula_matches_contract():
    """priority = posterior × parse_delta_mid(delta) / (recon_steps + run_minutes)。"""
    from knowledge_pipeline.simulator.scheduler import parse_delta_mid

    mw, _ = make_mw(graph=make_e9_graph())
    # H2: posterior 0.5 × parse_delta_mid("+0.05~0.15")=0.1 / (1+5)
    assert mw._priority("H2") == pytest.approx(0.5 * parse_delta_mid("+0.05~0.15") / 6)


# --------------------------------------------------------------------------
# d) 视图渲染
# --------------------------------------------------------------------------


def test_view_contains_four_content_classes_and_discipline():
    mw, store = make_mw()
    view = _render_view(mw, store)
    assert "[知识图谱 · meta-plan 视图]（第 1 次刷新）" in view
    assert "目标：" in view and "验收进展：" in view  # ① 目标与验收进展
    assert "假说全景" in view and "posterior" in view  # ② 假说全景与后验
    assert "冲突与约束：" in view  # ③ 冲突/冻结/备选组状态
    assert "待决决策点：" in view  # ④ 待决决策点
    assert "纪律：" in view and "kg_report_evidence" in view and "kg_decision" in view
    assert len(view) <= 2000


def test_view_refresh_count_and_persisted():
    mw, store = make_mw()
    _render_view(mw, store)
    _render_view(mw, store)
    view = _render_view(mw, store)
    assert "第 3 次刷新" in view
    assert _state_doc(store)["refresh_count"] == 3


def test_view_truncation_budget_2000():
    """假说列表超预算 → 按 E9 排序截断，已终局只保留计数，整体 ≤2000。"""
    graph = make_e9_graph()
    for i in range(7, 40):
        h = _hyp(f"H{i}", 0.5, "+0.10")
        h["mechanism"] = f"H{i} 的长机制描述" * 10
        graph["nodes"].append(h)
    mw, store = make_mw(graph=graph)
    mw.kg_report_evidence("H4", "confirmed", "e")
    view = _render_view(mw, store)
    assert len(view) <= 2000
    assert "因视图预算未列出" in view
    assert "已终局 1 条" in view  # 已终局只保留计数


def test_view_and_tool_texts_have_no_recommendation_wording():
    """用户拍板（D2/D5 定位修正）：信息供给而非路线推荐——禁用"推荐/建议"措辞。"""
    mw, store = make_mw(graph=make_e9_graph())
    texts = [
        _render_view(mw, store),
        mw.kg_next(),
        mw.kg_report_evidence("H4", "refuted", "e"),
        mw.kg_report_evidence("H1", "blocked", "e"),
        _render_view(mw, store),  # 含 boost/冻结标注的视图
        mw.kg_decision("J1", "切换 H3", "理由充分"),
        mw.kg_decision("J1", "x", ""),  # 拒绝文案同样约束
    ]
    for text in texts:
        for word in BANNED_WORDS:
            assert word not in text, f"措辞违规 {word!r}: {text[:120]}..."


def test_view_shows_frozen_and_boost_as_graph_structure_hints():
    mw, store = make_mw(graph=make_e9_graph())
    mw.kg_report_evidence("H4", "refuted", "e")
    mw.kg_report_evidence("H1", "blocked", "e")
    view = _render_view(mw, store)
    assert "兄弟假说 H4 已 refuted" in view
    assert "冻结中" in view


# --------------------------------------------------------------------------
# e) 启动接线（config 节 + Planner 挂载）
# --------------------------------------------------------------------------


def _kg_cfg(monkeypatch, cfg):
    monkeypatch.setitem(global_config, "knowledge_graph", cfg)


@pytest.fixture
def make_planner(monkeypatch):
    """Planner 构建器工厂：mock LLM（GenericFakeChatModel，不依赖真实端点/凭证）。"""
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

    from agent.factory import BaseAgentBuilder

    monkeypatch.setattr(
        BaseAgentBuilder, "_get_llm", lambda self: GenericFakeChatModel(messages=iter([]))
    )

    def _make(**kwargs):
        return PlannerAgentBuilder(
            llm_name="dsv4", store=InMemoryStore(), checkpointer=InMemorySaver(), **kwargs
        )

    return _make


def test_prepare_kg_disabled_by_default(monkeypatch):
    monkeypatch.delitem(global_config, "knowledge_graph", raising=False)
    assert AutoTuneAgent()._prepare_knowledge_graph() is None
    _kg_cfg(monkeypatch, {"enabled": False})
    assert AutoTuneAgent()._prepare_knowledge_graph() is None


def test_prepare_kg_valid_graph_mounted(monkeypatch, make_planner):
    _kg_cfg(monkeypatch, {"enabled": True, "graph_path": str(GOLDEN), "uncertain_dampen": 0.5})
    ctx = AutoTuneAgent()._prepare_knowledge_graph()
    assert ctx is not None and ctx["graph_path"] == str(GOLDEN)
    assert ctx["graph"]["nodes"]  # 图内容已加载

    middleware = make_planner(kg_context=ctx)._get_middleware()
    assert len(middleware) == 4  # 尾部 append
    assert isinstance(middleware[-1], KnowledgeGraphMiddleware)
    assert middleware[-1].uncertain_dampen == 0.5


def test_prepare_kg_invalid_abort(monkeypatch, tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("version: '1.0'\nnodes: []\n", encoding="utf-8")
    _kg_cfg(monkeypatch, {"enabled": True, "graph_path": str(bad), "on_invalid": "abort"})
    with pytest.raises(RuntimeError, match="M1 校验"):
        AutoTuneAgent()._prepare_knowledge_graph()


def test_prepare_kg_invalid_warn_disable(monkeypatch, tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("version: '1.0'\nnodes: []\n", encoding="utf-8")
    _kg_cfg(monkeypatch, {"enabled": True, "graph_path": str(bad), "on_invalid": "warn_disable"})
    assert AutoTuneAgent()._prepare_knowledge_graph() is None  # 等价未开启


def test_prepare_kg_missing_file_abort(monkeypatch, tmp_path):
    _kg_cfg(
        monkeypatch,
        {"enabled": True, "graph_path": str(tmp_path / "nope.yaml"), "on_invalid": "abort"},
    )
    with pytest.raises(RuntimeError):
        AutoTuneAgent()._prepare_knowledge_graph()


def test_planner_middleware_list_unchanged_when_disabled(make_planner):
    """enabled 缺省/false（kg_context=None）时 _get_middleware 返回列表与现状完全相同。"""
    middleware = make_planner()._get_middleware()
    assert [type(m) for m in middleware] == [
        SummarizationMiddleware,
        ModelRetryMiddleware,
        _ToolExclusionMiddleware,
    ]
    assert len(middleware) == 3


# --------------------------------------------------------------------------
# f) 导出（finally 阶段）
# --------------------------------------------------------------------------


def test_export_kg_log_two_files_schema_valid(tmp_path):
    agent = AutoTuneAgent()
    state_doc = {
        "version": "1.0",
        "graph_path": "kg/graph.yaml",
        "nodes": {"H1": {"state": "confirmed", "posterior": 0.95}, "A2": {"state": "confirmed"}},
        "decisions": {"J2": {"branch": "阈值+K折升级", "reason": "r", "ts": "2026-01-01T00:00:00+00:00"}},
        "gain_history": {"A6": [0.03, 0.01]},
        "refresh_count": 7,
    }
    record = {
        "node_id": "H1",
        "status": "confirmed",
        "evidence": "fit 调用缺参",
        "metrics": {"gain": 0.12},
        "posterior": 0.95,
        "notes": "",
        "ts": "2026-01-01T00:00:00+00:00",
    }
    agent.store.put(KG_NAMESPACE, KG_STATE_KEY, create_file_data(json.dumps(state_doc, ensure_ascii=False)))
    agent.store.put(KG_NAMESPACE, KG_LEDGER_KEY, create_file_data(json.dumps(record, ensure_ascii=False)))

    agent._export_kg_log(str(tmp_path))

    exported_state = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    for key in ("version", "graph_path", "nodes", "decisions", "gain_history", "refresh_count"):
        assert key in exported_state
    lines = (tmp_path / "hypotheses.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    Ledger.validate_record(json.loads(lines[0]))  # 账本 schema 复用 M3 校验


def test_export_kg_log_end_to_end_from_middleware(tmp_path):
    """middleware 写入 → finally 导出：两文件齐全且 schema 合法。"""
    mw, store = make_mw()
    mw.kg_report_evidence("H1", "confirmed", "fit 调用缺参", measured_gain=0.12)

    agent = AutoTuneAgent()
    agent.store = store  # 主系统同一 store
    agent._export_kg_log(str(tmp_path))

    exported_state = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert exported_state["nodes"]["H1"]["posterior"] == 0.95
    for key in ("version", "graph_path", "nodes", "decisions", "gain_history", "refresh_count"):
        assert key in exported_state
    lines = (tmp_path / "hypotheses.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    Ledger.validate_record(json.loads(lines[0]))


def test_export_kg_log_no_kg_data_is_graceful(tmp_path):
    """知识图谱未开启：导出静默跳过、不建文件、不抛异常（不掩盖主流程结果）。"""
    agent = AutoTuneAgent()
    agent._export_kg_log(str(tmp_path))
    assert not (tmp_path / "state.json").exists()
    assert not (tmp_path / "hypotheses.jsonl").exists()
