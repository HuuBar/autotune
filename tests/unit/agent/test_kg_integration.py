"""W-C 知识图谱集成测试（设计书 §7 验收 V2/V3/V6/V7，离线可落地部分）。

与 W-B 单测（test_kg_middleware.py，组件级）的区别：本文件走真实构建链——
`AutoTuneAgent._prepare_knowledge_graph()`（config 节 → M1 校验 → kg_context）
→ `PlannerAgentBuilder._get_middleware()`（条件 append）
→ `PlannerAgentBuilder.build()`（真实 create_deep_agent）
→ `agent.invoke(...)`（真实 middleware 链 + 真实工具调用回路）
→ `AutoTuneAgent._export_kg_log()`（finally 导出）。
LLM 全程用 ScriptedChatModel（langchain_core BaseChatModel 子类，录制收到的
消息并按脚本返回），零真实端点、零凭证。

覆盖（§7）：
- V2 关闭态：enabled 缺省/false → Planner middleware 列表与现状逐类型相同；
  走一轮 agent 调用后 state 无 /memories/kg/ 写入、日志无 kg 痕迹、模型收到的
  system_message 无 meta-plan 视图。
- V3 开启态启动：enabled:true + tbox_golden 合法图 → 正常启动（kg_context 非
  None、middleware 尾部挂载）；Planner 首轮模型调用收到的 system_message 中
  可见 meta-plan 视图标志性行。
- V6 降级正确：bad_graphs 非法图 + warn_disable → 记 warning 且行为等价关闭
  （kg_context=None、middleware 不挂载）；abort → RuntimeError，信息含校验错误。
- V7 导出齐全：跑一轮（含一次真实工具调用回路的 kg_report_evidence）后触发
  导出 → 目标目录 state.json + hypotheses.jsonl 齐全；state.json 过 §4.4
  schema 关键字段；hypotheses.jsonl 每行过 M3 Ledger.validate_record。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from agent.agent import AutoTuneAgent
from agent.factory import BaseAgentBuilder, PlannerAgentBuilder
from agent.kg_middleware import KG_LEDGER_KEY, KG_NAMESPACE, KG_STATE_KEY, KnowledgeGraphMiddleware
from deepagents.middleware._tool_exclusion import _ToolExclusionMiddleware
from knowledge_pipeline import load_graph
from knowledge_pipeline.ledger import Ledger
from langchain.agents.middleware import ModelRetryMiddleware, SummarizationMiddleware
from utils.config import global_config
from utils.logger import logger

FIXTURES = Path(__file__).resolve().parents[3] / "knowledge_pipeline" / "fixtures"
GOLDEN = FIXTURES / "tbox_golden.yaml"
BAD_GRAPHS = FIXTURES / "bad_graphs"

VIEW_MARK = "[知识图谱 · meta-plan 视图]"

#: 关闭态下 Planner middleware 现状（锚点 A2，逐类型比对）
BASELINE_MIDDLEWARE_TYPES = [SummarizationMiddleware, ModelRetryMiddleware, _ToolExclusionMiddleware]


# --------------------------------------------------------------------------
# 测试夹具：脚本化假模型（录制收到的消息；零真实端点）
# --------------------------------------------------------------------------


class ScriptedChatModel(BaseChatModel):
    """录制每次模型调用收到的 messages，并按脚本返回 AIMessage。

    脚本规则：
    - scripted_tool_call 非 None 且会话中尚无 ToolMessage → 返回携带该
      tool_call 的 AIMessage（驱动真实工具调用回路，V7 用）；
    - 否则返回普通文本 AIMessage（agent 正常收尾）。
    """

    seen: list = []
    scripted_tool_call: dict | None = None

    @property
    def _llm_type(self) -> str:
        return "scripted-fake-chat-model"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedChatModel":
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.seen.append(list(messages))
        if self.scripted_tool_call and not any(isinstance(m, ToolMessage) for m in messages):
            ai = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": self.scripted_tool_call["name"],
                        "args": self.scripted_tool_call["args"],
                        "id": "call_kg_1",
                        "type": "tool_call",
                    }
                ],
            )
        else:
            ai = AIMessage(content="收到，任务完成。")
        return ChatResult(generations=[ChatGeneration(message=ai)])


@pytest.fixture
def fake_llm_factory(monkeypatch):
    """mock BaseAgentBuilder._get_llm：返回 ScriptedChatModel 并登记，供断言回放。"""

    def _factory(scripted_tool_call: dict | None = None):
        created: list[ScriptedChatModel] = []

        def _get_llm(self):
            model = ScriptedChatModel(scripted_tool_call=scripted_tool_call)
            created.append(model)
            return model

        monkeypatch.setattr(BaseAgentBuilder, "_get_llm", _get_llm)
        return created

    return _factory


@pytest.fixture
def autotune_log():
    """捕获 utils.logger（propagate=False，caplog 捕获不到）的日志记录。"""

    class _ListHandler(logging.Handler):
        def __init__(self):
            super().__init__()
            self.messages: list[str] = []

        def emit(self, record: logging.LogRecord) -> None:
            self.messages.append(record.getMessage())

    handler = _ListHandler()
    logger.addHandler(handler)
    try:
        yield handler
    finally:
        logger.removeHandler(handler)


def _kg_cfg(monkeypatch, cfg: dict | None):
    if cfg is None:
        monkeypatch.delitem(global_config, "knowledge_graph", raising=False)
    else:
        monkeypatch.setitem(global_config, "knowledge_graph", cfg)


def _make_planner(store: InMemoryStore, kg_context: dict | None) -> PlannerAgentBuilder:
    return PlannerAgentBuilder(
        llm_name="dsv4", store=store, checkpointer=InMemorySaver(), kg_context=kg_context
    )


def _invoke_one_round(builder: PlannerAgentBuilder) -> None:
    """真实构建链 + 走一轮 agent 调用（构建失败/调用异常即测试失败）。"""
    agent = builder.build()
    agent.invoke(
        {"messages": [{"role": "user", "content": "请开始规划 TBox 任务。"}]},
        config={"configurable": {"thread_id": "kg-it"}},
    )


def _all_seen_messages(models: list[ScriptedChatModel]) -> list[list[BaseMessage]]:
    return [call for model in models for call in model.seen]


def _system_texts(calls: list[list[BaseMessage]]) -> list[str]:
    texts = []
    for messages in calls:
        for m in messages:
            if m.type == "system":
                texts.append(m.content if isinstance(m.content, str) else str(m.content))
    return texts


# --------------------------------------------------------------------------
# V2 关闭态：行为与现状一致，零 kg 痕迹
# --------------------------------------------------------------------------


@pytest.mark.parametrize("cfg", [None, {"enabled": False}], ids=["缺省", "enabled:false"])
def test_v2_disabled_middleware_and_run_leave_no_kg_trace(
    monkeypatch, fake_llm_factory, autotune_log, cfg
):
    _kg_cfg(monkeypatch, cfg)
    models = fake_llm_factory()

    # 启动接线：缺省/false → kg_context 为 None
    kg_context = AutoTuneAgent()._prepare_knowledge_graph()
    assert kg_context is None

    # Planner middleware 列表与现状逐类型相同（锚点 A2）
    store = InMemoryStore()
    middleware = _make_planner(store, kg_context)._get_middleware()
    assert [type(m) for m in middleware] == BASELINE_MIDDLEWARE_TYPES

    # 走一轮 agent 调用后：无 /memories/kg/ 写入、模型未见视图、日志无 kg 痕迹
    _invoke_one_round(_make_planner(store, kg_context))
    assert models and _all_seen_messages(models), "fake 模型未被调用（构建链未走通）"
    assert store.get(KG_NAMESPACE, KG_STATE_KEY) is None
    assert store.get(KG_NAMESPACE, KG_LEDGER_KEY) is None
    for text in _system_texts(_all_seen_messages(models)):
        assert VIEW_MARK not in text
    assert not any("知识图谱" in msg or "/memories/kg" in msg for msg in autotune_log.messages)


def test_v2_disabled_export_creates_nothing(monkeypatch, tmp_path):
    """关闭态 finally 导出语义：无 kg 状态 → 不建文件、不抛异常。"""
    _kg_cfg(monkeypatch, {"enabled": False})
    AutoTuneAgent()._export_kg_log(str(tmp_path / "kg"))
    assert not (tmp_path / "kg").exists() or list((tmp_path / "kg").iterdir()) == []


# --------------------------------------------------------------------------
# V3 开启态启动：合法图 → 挂载；Planner 首轮模型调用可见 meta-plan 视图
# --------------------------------------------------------------------------


def test_v3_enabled_valid_graph_mounts_and_view_reaches_planner(
    monkeypatch, fake_llm_factory, autotune_log
):
    _kg_cfg(monkeypatch, {"enabled": True, "graph_path": str(GOLDEN), "uncertain_dampen": 0.5})
    models = fake_llm_factory()

    # 正常启动：M1 校验通过 → kg_context 非 None
    kg_context = AutoTuneAgent()._prepare_knowledge_graph()
    assert kg_context is not None
    assert kg_context["graph"]["nodes"] and kg_context["graph_path"] == str(GOLDEN)
    assert any("知识图谱已加载" in msg for msg in autotune_log.messages)

    # Planner middleware 尾部挂载 KnowledgeGraphMiddleware（前 3 项与现状一致）
    store = InMemoryStore()
    middleware = _make_planner(store, kg_context)._get_middleware()
    assert [type(m) for m in middleware[:3]] == BASELINE_MIDDLEWARE_TYPES
    assert isinstance(middleware[-1], KnowledgeGraphMiddleware)

    # 走一轮：Planner 首轮模型调用收到的 system_message 中可见 meta-plan 视图
    _invoke_one_round(_make_planner(store, kg_context))
    calls = _all_seen_messages(models)
    assert calls, "fake 模型未被调用"
    first_call_system = _system_texts(calls[:1])
    assert first_call_system, "首轮调用无 system_message"
    first_view = first_call_system[0]
    assert VIEW_MARK in first_view  # 视图标志性行
    assert "（第 1 次刷新）" in first_view
    assert "假说全景" in first_view and "验收进展：" in first_view
    assert "纪律：" in first_view  # §4.3 四类内容 + 纪律行
    # 视图状态已落 /memories/kg/（refresh_count ≥ 1）
    item = store.get(KG_NAMESPACE, KG_STATE_KEY)
    assert item, "state.json 未落盘"
    assert json.loads(item.value["content"])["refresh_count"] >= 1


# --------------------------------------------------------------------------
# V6 降级正确：非法图 + on_invalid 双策略
# --------------------------------------------------------------------------

BAD_GRAPH_FILES = sorted(BAD_GRAPHS.glob("bad_*.yaml"))


def test_v6_bad_fixtures_are_actually_invalid():
    """夹具自检：bad_graphs 全部未过 M1 校验（防夹具漂移导致 V6 假阳性）。"""
    from knowledge_pipeline.validator import validate_file

    assert BAD_GRAPH_FILES, "bad_graphs 夹具缺失"
    for path in BAD_GRAPH_FILES:
        assert validate_file(path), f"{path.name} 意外通过了 M1 校验"


@pytest.mark.parametrize("bad_graph", BAD_GRAPH_FILES, ids=lambda p: p.name)
def test_v6_invalid_graph_warn_disable_equivalent_to_off(
    monkeypatch, fake_llm_factory, autotune_log, bad_graph
):
    _kg_cfg(
        monkeypatch,
        {"enabled": True, "graph_path": str(bad_graph), "on_invalid": "warn_disable"},
    )
    models = fake_llm_factory()

    # 记 warning 且等价关闭（kg_context=None）
    kg_context = AutoTuneAgent()._prepare_knowledge_graph()
    assert kg_context is None
    warnings = [m for m in autotune_log.messages if "M1 校验" in m and "warn_disable" in m]
    assert warnings, "未记录 warn_disable 降级的 warning"

    # 行为等价关闭：middleware 不挂载、走一轮无 kg 痕迹
    store = InMemoryStore()
    middleware = _make_planner(store, kg_context)._get_middleware()
    assert [type(m) for m in middleware] == BASELINE_MIDDLEWARE_TYPES
    _invoke_one_round(_make_planner(store, kg_context))
    assert store.get(KG_NAMESPACE, KG_STATE_KEY) is None
    for text in _system_texts(_all_seen_messages(models)):
        assert VIEW_MARK not in text


@pytest.mark.parametrize("bad_graph", BAD_GRAPH_FILES[:2], ids=lambda p: p.name)
def test_v6_invalid_graph_abort_raises_with_validation_errors(monkeypatch, bad_graph):
    _kg_cfg(
        monkeypatch,
        {"enabled": True, "graph_path": str(bad_graph), "on_invalid": "abort"},
    )
    with pytest.raises(RuntimeError) as excinfo:
        AutoTuneAgent()._prepare_knowledge_graph()
    message = str(excinfo.value)
    assert "M1 校验" in message
    assert bad_graph.name in message or str(bad_graph) in message
    # 报错信息含具体校验错误（fail-fast，不允许静默通过）
    from knowledge_pipeline.validator import validate_file

    assert any(str(e) in message for e in validate_file(bad_graph))


def test_v6_abort_is_default_policy(monkeypatch):
    """on_invalid 缺省 = abort（设计书 §4.1）。"""
    _kg_cfg(
        monkeypatch,
        {"enabled": True, "graph_path": str(BAD_GRAPH_FILES[0])},
    )
    with pytest.raises(RuntimeError, match="M1 校验"):
        AutoTuneAgent()._prepare_knowledge_graph()


# --------------------------------------------------------------------------
# V7 导出齐全：真实工具调用回路 → finally 导出 → schema 校验
# --------------------------------------------------------------------------


def test_v7_export_state_and_ledger_complete_after_one_round(
    monkeypatch, fake_llm_factory, tmp_path
):
    _kg_cfg(monkeypatch, {"enabled": True, "graph_path": str(GOLDEN)})
    # 脚本：首轮对 kg_report_evidence 发起真实工具调用（H1 confirmed，gain 0.12）
    models = fake_llm_factory(
        scripted_tool_call={
            "name": "kg_report_evidence",
            "args": {
                "node_id": "H1",
                "status": "confirmed",
                "evidence": "fit 调用缺 app_type_series，见 train.py:88",
                "measured_gain": 0.12,
            },
        }
    )

    kg_context = AutoTuneAgent()._prepare_knowledge_graph()
    assert kg_context is not None
    store = InMemoryStore()
    _invoke_one_round(_make_planner(store, kg_context))

    # 工具确实经真实调用回路执行过（账本有一行 H1 confirmed）
    ledger_item = store.get(KG_NAMESPACE, KG_LEDGER_KEY)
    assert ledger_item, "kg_report_evidence 未执行（账本为空）"
    store_lines = [l for l in ledger_item.value["content"].splitlines() if l.strip()]
    assert len(store_lines) == 1
    assert json.loads(store_lines[0])["node_id"] == "H1"

    # 触发导出（仿 agent.py finally：目标目录 kg/）
    agent = AutoTuneAgent()
    agent.store = store  # 与运行期同一 store
    export_dir = tmp_path / "kg"
    agent._export_kg_log(str(export_dir))

    # state.json + hypotheses.jsonl 齐全
    state_path, ledger_path = export_dir / "state.json", export_dir / "hypotheses.jsonl"
    assert state_path.is_file() and ledger_path.is_file()

    # state.json 过 §4.4 schema 关键字段检查
    state = json.loads(state_path.read_text(encoding="utf-8"))
    for key in ("version", "graph_path", "nodes", "decisions", "gain_history", "refresh_count"):
        assert key in state, f"state.json 缺 §4.4 关键字段 {key}"
    assert state["version"] == "1.0"
    assert state["graph_path"] == str(GOLDEN)
    assert state["nodes"]["H1"] == {"state": "confirmed", "posterior": 0.95}
    assert state["gain_history"]["H1"] == [0.12]
    assert state["refresh_count"] >= 1  # wrap_model_call 真实执行过

    # hypotheses.jsonl 每行过 M3 Ledger.validate_record
    lines = [l for l in ledger_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 1
    for line in lines:
        Ledger.validate_record(json.loads(line))
    record = json.loads(lines[0])
    assert record["node_id"] == "H1" and record["status"] == "confirmed"
    assert record["posterior"] == 0.95 and record["metrics"] == {"gain": 0.12}
    assert models and _all_seen_messages(models)
