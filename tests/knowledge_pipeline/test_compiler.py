"""M4 编排编译器测试。

覆盖（SPEC §6 / 任务书验收 a~f）：
a) FakeClient 走通全流程（选卡 → 成图 → 过校验）；
b) 校验门拦截：先坏图（缺 verify）→ 回灌错误重试 → 好图，断言重试发生且终过；
c) 超 max_retries 仍坏 → CompileError（不绕过校验）；
d) 编译器自检拦截：领域 <3、假说 mechanism 为空、card_decisions 未覆盖全卡库；
e) CachedClient 用预录 TBox cache 端到端产出过校验的图；
f) OpenAICompatClient 端点注入（URL/头/体由配置构造，mock urlopen，不发真实网络）。
另：鲁棒解析（围栏 + 首尾杂讯）在 a/e 中间接覆盖，并有专项用例。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from knowledge_pipeline import load_graph, validate_graph
from knowledge_pipeline.compiler import (
    CARD_SELECTION_SYSTEM,
    GRAPH_SYSTEM,
    CachedClient,
    CompileError,
    FakeClient,
    OpenAICompatClient,
    card_selection_user,
    compile_graph,
    graph_user,
)
from knowledge_pipeline.compiler.llm_client import LLMResponseError

FIXTURES = Path(__file__).resolve().parents[2] / "knowledge_pipeline" / "fixtures"
CARDS_MD = (
    Path(__file__).resolve().parents[2]
    / "docs" / "knowledge-pipeline" / "自动化ML优化战术知识卡_v1.md"
)
BRIEF = FIXTURES / "tbox_brief.yaml"
LLM_CACHE = FIXTURES / "llm_cache"
GOLDEN = FIXTURES / "tbox_golden.yaml"

SELECT_KEY = "【阶段：选卡】"
GRAPH_KEY = "【阶段：成图】"


# --------------------------------------------------------------------------
# 小型夹具（tmp_path）：3 张卡 + 最小合法图
# --------------------------------------------------------------------------

MINI_CARDS = """# 测试卡库

### X1 接线审计卡 [动作]
**做法**：调参前先验证机制真接线。

### X2 不适用卡 [决策]
**做法**：多轨迹并行探索。

### X3 冒烟先行卡 [动作]
**做法**：小样本先行，全量殿后。
"""

MINI_BRIEF = """brief:
  task_id: "mini"
  goal:
    statement: "mini 指标 >= 0.5"
    acceptance:
      - "m.a >= 0.5"
terrain:
  status: "未知，待侦查"
"""

MINI_DECISIONS = [
    {"card_id": "X1", "decision": "adopt", "rationale": "存量代码库先审计接线，适用"},
    {"card_id": "X2", "decision": "reject", "rationale": "v1 串行调度，不引入并行轨迹"},
    {"card_id": "X3", "decision": "adopt", "rationale": "edit 动作先冒烟，拦低级错误"},
]

GOOD_GRAPH = """\
version: "1.0"
meta:
  task: "mini"
  created_by: "fake-llm"
  source_cards: ["X1", "X3"]
goal:
  statement: "mini 指标 >= 0.5"
  acceptance:
    - "m.a >= 0.5"
nodes:
  - {id: D1, layer: domain, rationale: "卡片 X1：接线高频失败"}
  - {id: D2, layer: domain, rationale: "任务书基线不平衡"}
  - {id: D3, layer: domain, rationale: "存在零重要度特征"}
  - id: H1
    layer: hypothesis
    domain: D1
    mechanism: "未接线的机制从未生效，接通即无代价收益"
    prior: 0.5
    expected: {metric: "m.a", delta: "+0.1", confidence: "low"}
    cost: {recon_steps: 2, run_minutes: 5}
  - id: A1
    layer: action
    belongs_to: H1
    type: recon
    do: "审计调用链接线"
    tool_hint: "read_code_block"
    produces: "wiring_report"
    verify: "每个机制有明确接线判定"
    retry_budget: 1
edges: []
decisions: []
writeback:
  ledger: "hypotheses.jsonl"
  record_schema: "{node_id, status, evidence, metrics, posterior, notes, ts}"
  after_domain_review: "按领域聚合"
"""

# 坏图 1：action 缺 verify（触发 V-ACTION-VERIFY，M1 校验门拦截）
BAD_GRAPH_NO_VERIFY = GOOD_GRAPH.replace(
    '    verify: "每个机制有明确接线判定"\n', ""
)

# 坏图 2：仅 2 个领域（M1 校验全绿，编译器自检拦截）
BAD_GRAPH_TWO_DOMAINS = GOOD_GRAPH.replace(
    '  - {id: D3, layer: domain, rationale: "存在零重要度特征"}\n', ""
)

# 坏图 3：hypothesis mechanism 为空字符串（M1 只看字段存在，编译器自检拦截）
BAD_GRAPH_EMPTY_MECHANISM = GOOD_GRAPH.replace(
    'mechanism: "未接线的机制从未生效，接通即无代价收益"', 'mechanism: ""'
)


@pytest.fixture()
def mini_inputs(tmp_path: Path) -> tuple[Path, Path]:
    cards = tmp_path / "cards.md"
    cards.write_text(MINI_CARDS, encoding="utf-8")
    brief = tmp_path / "brief.yaml"
    brief.write_text(MINI_BRIEF, encoding="utf-8")
    return cards, brief


def select_payload(decisions: list[dict] = MINI_DECISIONS) -> str:
    # 带 ```json 围栏 + 首尾杂讯，同时覆盖鲁棒解析
    return "好的，以下是论证表：\n```json\n" + json.dumps(
        decisions, ensure_ascii=False
    ) + "\n```\n以上。"


def graph_payload(graph_yaml: str = GOOD_GRAPH) -> str:
    return "编排图如下。\n```yaml\n" + graph_yaml + "\n```\n（完）"


# --------------------------------------------------------------------------
# a) FakeClient 全流程
# --------------------------------------------------------------------------


def test_full_flow_with_fake_client(mini_inputs: tuple[Path, Path]) -> None:
    cards, brief = mini_inputs
    client = FakeClient({SELECT_KEY: select_payload(), GRAPH_KEY: graph_payload()})
    result = compile_graph(cards, brief, client)

    assert result.attempts == 1
    assert result.select_attempts == 1
    assert validate_graph(result.graph) == []          # 过 M1 确定性校验门
    assert yaml.safe_load(result.graph_yaml) == result.graph
    assert [d["card_id"] for d in result.card_decisions] == ["X1", "X2", "X3"]
    assert all(d["rationale"] for d in result.card_decisions)
    assert len(client.calls) == 2                       # 选卡 1 次 + 成图 1 次


def test_result_is_llm_client_protocol(mini_inputs: tuple[Path, Path]) -> None:
    from knowledge_pipeline.compiler import LLMClient

    client = FakeClient({SELECT_KEY: select_payload(), GRAPH_KEY: graph_payload()})
    assert isinstance(client, LLMClient)


# --------------------------------------------------------------------------
# b) 校验门拦截：坏图 → 错误回灌重试 → 好图
# --------------------------------------------------------------------------


def test_validation_gate_retries_with_error_feedback(
    mini_inputs: tuple[Path, Path],
) -> None:
    cards, brief = mini_inputs
    client = FakeClient(
        {
            SELECT_KEY: select_payload(),
            GRAPH_KEY: [graph_payload(BAD_GRAPH_NO_VERIFY), graph_payload(GOOD_GRAPH)],
        }
    )
    result = compile_graph(cards, brief, client)

    assert result.attempts == 2                          # 重试确实发生
    assert validate_graph(result.graph) == []
    # 第二次成图调用的 prompt 中应回灌了 V-ACTION-VERIFY 错误
    second_graph_prompt = client.calls[2][1]
    assert "V-ACTION-VERIFY" in second_graph_prompt
    assert "verify" in second_graph_prompt


# --------------------------------------------------------------------------
# c) 超 max_retries 仍坏 → CompileError（不绕过校验）
# --------------------------------------------------------------------------


def test_exceeding_max_retries_raises_compile_error(
    mini_inputs: tuple[Path, Path],
) -> None:
    cards, brief = mini_inputs
    client = FakeClient(
        {SELECT_KEY: select_payload(), GRAPH_KEY: graph_payload(BAD_GRAPH_NO_VERIFY)}
    )
    with pytest.raises(CompileError) as excinfo:
        compile_graph(cards, brief, client, max_retries=2)

    err = excinfo.value
    assert err.stage == "成图"
    assert err.attempts == 2
    assert any("V-ACTION-VERIFY" in p for p in err.problems)
    assert len(client.calls) == 1 + 2                    # 选卡 1 + 成图 2（不多调）


def test_unparseable_output_counts_as_failed_attempt(
    mini_inputs: tuple[Path, Path],
) -> None:
    cards, brief = mini_inputs
    client = FakeClient(
        {SELECT_KEY: select_payload(), GRAPH_KEY: ["这不是 YAML：{{{", graph_payload()]}
    )
    result = compile_graph(cards, brief, client)
    assert result.attempts == 2                          # 解析失败按一次失败重试处理


# --------------------------------------------------------------------------
# d) 编译器自检拦截
# --------------------------------------------------------------------------


def test_self_check_rejects_fewer_than_3_domains(
    mini_inputs: tuple[Path, Path],
) -> None:
    cards, brief = mini_inputs
    client = FakeClient(
        {
            SELECT_KEY: select_payload(),
            GRAPH_KEY: [graph_payload(BAD_GRAPH_TWO_DOMAINS), graph_payload()],
        }
    )
    result = compile_graph(cards, brief, client)
    assert result.attempts == 2
    assert "领域" in client.calls[2][1]                  # 自检问题已回灌


def test_self_check_rejects_empty_mechanism(
    mini_inputs: tuple[Path, Path],
) -> None:
    cards, brief = mini_inputs
    client = FakeClient(
        {
            SELECT_KEY: select_payload(),
            GRAPH_KEY: [graph_payload(BAD_GRAPH_EMPTY_MECHANISM), graph_payload()],
        }
    )
    result = compile_graph(cards, brief, client)
    assert result.attempts == 2
    assert "mechanism" in client.calls[2][1]


def test_self_check_rejects_incomplete_card_coverage(
    mini_inputs: tuple[Path, Path],
) -> None:
    cards, brief = mini_inputs
    partial = [d for d in MINI_DECISIONS if d["card_id"] != "X3"]  # 漏一张卡
    client = FakeClient(
        {
            SELECT_KEY: [select_payload(partial), select_payload()],
            GRAPH_KEY: graph_payload(),
        }
    )
    result = compile_graph(cards, brief, client)
    assert result.select_attempts == 2                   # 选卡步触发重试
    assert result.attempts == 1
    assert {d["card_id"] for d in result.card_decisions} == {"X1", "X2", "X3"}
    assert "X3" in client.calls[1][1]                    # 缺失卡 id 已回灌


def test_self_check_rejects_empty_rationale(
    mini_inputs: tuple[Path, Path],
) -> None:
    cards, brief = mini_inputs
    no_rationale = [
        {"card_id": "X1", "decision": "adopt", "rationale": ""},
        *MINI_DECISIONS[1:],
    ]
    client = FakeClient(
        {SELECT_KEY: select_payload(no_rationale), GRAPH_KEY: graph_payload()}
    )
    with pytest.raises(CompileError) as excinfo:
        compile_graph(cards, brief, client, max_retries=1)
    assert excinfo.value.stage == "选卡"
    assert any("rationale" in p for p in excinfo.value.problems)


# --------------------------------------------------------------------------
# e) CachedClient + 预录 TBox cache 端到端
# --------------------------------------------------------------------------


def test_cached_client_tbox_end_to_end() -> None:
    client = CachedClient(LLM_CACHE)
    result = compile_graph(CARDS_MD, BRIEF, client)

    assert validate_graph(result.graph) == []            # 过 M1 确定性校验门
    # card_decisions 覆盖卡库全部 29 张卡，adopt/reject 均有 rationale
    assert len(result.card_decisions) == 29
    assert all(d["decision"] in ("adopt", "reject") for d in result.card_decisions)
    assert all(d["rationale"] for d in result.card_decisions)
    # 采用卡与黄金图 meta.source_cards 一致（黄金图即 manual v1 编译产物）
    adopted = {d["card_id"] for d in result.card_decisions if d["decision"] == "adopt"}
    golden = load_graph(GOLDEN)
    assert adopted == set(golden["meta"]["source_cards"])
    # 产出图与黄金图同构（节点/边/决策 id 集合一致）
    same = lambda key: {n["id"] for n in result.graph[key]} == {n["id"] for n in golden[key]}
    assert same("nodes") and same("decisions")


def test_cached_client_cache_miss_raises(tmp_path: Path) -> None:
    client = CachedClient(tmp_path)
    with pytest.raises(KeyError, match="cache 未命中"):
        client.complete("s", "u")


# --------------------------------------------------------------------------
# f) OpenAICompatClient 端点注入（不发真实网络）
# --------------------------------------------------------------------------


def test_openai_compat_explicit_config() -> None:
    client = OpenAICompatClient(
        base_url="http://llm.local:8000/v1/", model="kp-model", api_key="sk-test"
    )
    assert client.endpoint == "http://llm.local:8000/v1/chat/completions"  # 尾斜杠规整
    req = client._build_request("sys", "usr")
    assert req.full_url == client.endpoint
    assert req.get_method() == "POST"
    assert req.headers["Authorization"] == "Bearer sk-test"
    assert req.headers["Content-type"] == "application/json"
    body = json.loads(req.data.decode("utf-8"))
    assert body["model"] == "kp-model"
    assert body["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "usr"},
    ]


def test_openai_compat_env_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KP_LLM_BASE_URL", "http://env-llm:9000/v1")
    monkeypatch.setenv("KP_LLM_MODEL", "env-model")
    monkeypatch.setenv("KP_LLM_API_KEY", "sk-env")
    client = OpenAICompatClient()
    assert client.endpoint == "http://env-llm:9000/v1/chat/completions"
    assert client.model == "env-model"
    assert client._build_request("s", "u").headers["Authorization"] == "Bearer sk-env"


def test_openai_compat_missing_config_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("KP_LLM_BASE_URL", "KP_LLM_MODEL", "KP_LLM_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(ValueError, match="base_url"):
        OpenAICompatClient()
    with pytest.raises(ValueError, match="model"):
        OpenAICompatClient(base_url="http://x/v1")


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._raw = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._raw

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None


def test_openai_compat_complete_via_mocked_urlopen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_urlopen(req: object, timeout: float = 0.0) -> _FakeResponse:
        captured["url"] = req.full_url  # type: ignore[attr-defined]
        captured["auth"] = req.headers.get("Authorization")  # type: ignore[attr-defined]
        captured["timeout"] = timeout
        return _FakeResponse(
            {"choices": [{"message": {"content": "编排图 YAML 原文"}}]}
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = OpenAICompatClient(
        base_url="http://llm.local/v1", model="m", api_key="k", timeout=30
    )
    assert client.complete("sys", "usr") == "编排图 YAML 原文"
    assert captured["url"] == "http://llm.local/v1/chat/completions"
    assert captured["auth"] == "Bearer k"
    assert captured["timeout"] == 30


def test_openai_compat_bad_response_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda req, timeout=0.0: _FakeResponse({"unexpected": True}),
    )
    client = OpenAICompatClient(base_url="http://llm.local/v1", model="m")
    with pytest.raises(LLMResponseError):
        client.complete("s", "u")


# --------------------------------------------------------------------------
# 提示词硬约束（输出格式契约存在性）
# --------------------------------------------------------------------------


def test_prompts_carry_format_constraints_and_stage_markers() -> None:
    assert "【阶段：选卡】" in CARD_SELECTION_SYSTEM and "JSON" in CARD_SELECTION_SYSTEM
    assert "【阶段：成图】" in GRAPH_SYSTEM and "YAML" in GRAPH_SYSTEM
    user_sel = card_selection_user("卡库", "任务书")
    user_graph = graph_user("卡库", "任务书", MINI_DECISIONS, errors=["[V-DAG] 有环"])
    assert "【阶段：选卡】" in user_sel
    assert "【阶段：成图】" in user_graph
    assert "[V-DAG] 有环" in user_graph                   # 错误回灌段进入 prompt
