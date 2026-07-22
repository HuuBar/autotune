"""M3：假说账本（hypotheses.jsonl）+ 后验更新 + 领域聚合报告。

语义来源：
- SPEC §5（接口契约与状态词表：confirmed|refuted|uncertain|blocked）；
- 01 文档 §5「执行后：回写与增殖」——指标是数据不是文本；append-only；
  未验证观察标"待验"，防记忆回授污染；
- 03 文档第三部分——回写实例（H1 confirmed posterior 0.5→0.95、H2 confirmed
  实测 +0.12、H3 边际提升复跑判 uncertain"不计入战果"）与领域聚合示例
  （接线域确认高收益→prior 上调；加权域边际→降权；特征域未探索→保留）。

工程默认值（SPEC 未覆盖处，列入回传报告「新发现」）：
1. 后验更新系数 0.9：confirmed → p+(1-p)*0.9；refuted → p*0.1；
   uncertain/blocked → 不变（SPEC §5 已固化，锚点 0.5 confirmed → 0.95）。
2. total_gain 提取口径：默认取 metrics["gain"]（可用 gain_key 参数改键），
   仅 **confirmed** 记录的数值型 gain 计入领域总收益——uncertain 按 03 文档
   纪律"不计入战果"，refuted/blocked 同样不产生战果。bool 不视为数值。
3. prior_suggestion 判定顺序：confirmed≥1 且 total_gain>0 → up；
   (refuted≥1 或 uncertain≥1) 且无正收益 → down；其余（含未探索）→ hold。
4. 节点→领域映射：domain 节点映射到自身；hypothesis 取其 domain 字段；
   action 沿 belongs_to 链回溯到 hypothesis/domain。无法映射的记录不计入
   任何领域（静默跳过，调用方可先行校验 node_id）。
5. 03 文档称账本"追加 5 条（…未验×2）"，但 SPEC §5 状态词表不含"未验"：
   裁决为未验证的假说**不落账本**（无记录），domain_report 通过图内
   hypothesis→domain 映射识别"未探索"领域并给 hold。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# SPEC §5：账本记录必填字段与状态词表（对齐 writeback.record_schema 扩展）
REQUIRED_FIELDS = ("node_id", "status", "evidence", "metrics", "posterior", "notes", "ts")
STATUSES = ("confirmed", "refuted", "uncertain", "blocked")

# 后验更新系数【工程默认值】：锚点 03 文档 H1 prior 0.5 confirmed → 0.95
CONFIRM_FACTOR = 0.9
REFUTED_FACTOR = 1 - CONFIRM_FACTOR  # 0.1，与 confirmed 对称

# total_gain 的默认实测收益键【工程默认值，见模块 docstring 口径 2】
DEFAULT_GAIN_KEY = "gain"


def update_posterior(prior: float, status: str) -> float:
    """按 SPEC §5 工程默认值更新假说后验置信度。

    confirmed → p + (1-p)*0.9；refuted → p*0.1；
    uncertain / blocked → 不变（uncertain 应在 notes 记"不计入战果"）。

    锚点：update_posterior(0.5, "confirmed") == 0.95（03 文档 H1）。
    非法 status 或 prior 越界 [0,1] 抛 ValueError。
    """
    if status not in STATUSES:
        raise ValueError(f"非法 status：{status!r}（词表：{STATUSES}）")
    if not isinstance(prior, (int, float)) or isinstance(prior, bool) or not 0.0 <= prior <= 1.0:
        raise ValueError(f"prior 必须是 [0,1] 数值，得到：{prior!r}")
    p = float(prior)
    if status == "confirmed":
        return p + (1.0 - p) * CONFIRM_FACTOR
    if status == "refuted":
        return p * REFUTED_FACTOR
    return p  # uncertain / blocked 不变


class Ledger:
    """hypotheses.jsonl 假说账本：append-only，非法写回拒绝落盘。

    语义来源：01 文档 §5（指标是数据不是文本；append-only 防记忆回授污染）。
    append 采用追加打开模式（"a"），不做全量重写；校验失败抛 ValueError
    且**不触碰文件**（先校验后打开）。
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    @staticmethod
    def validate_record(record: dict) -> None:
        """校验单条记录；非法抛 ValueError（SPEC §5：非法写回拒绝）。"""
        if not isinstance(record, dict):
            raise ValueError(f"账本记录必须是 dict，得到：{type(record).__name__}")
        missing = [k for k in REQUIRED_FIELDS if k not in record]
        if missing:
            raise ValueError(f"账本记录缺必填字段 {missing}（要求：{list(REQUIRED_FIELDS)}）")
        if record["status"] not in STATUSES:
            raise ValueError(f"非法 status：{record['status']!r}（词表：{STATUSES}）")
        if not isinstance(record["metrics"], dict):
            raise ValueError(
                f"metrics 必须为 dict（指标是数据不是文本），得到：{type(record['metrics']).__name__}"
            )
        if not isinstance(record["node_id"], str) or not record["node_id"]:
            raise ValueError(f"node_id 必须是非空字符串，得到：{record['node_id']!r}")
        posterior = record["posterior"]
        if not isinstance(posterior, (int, float)) or isinstance(posterior, bool) or not 0.0 <= posterior <= 1.0:
            raise ValueError(f"posterior 必须是 [0,1] 数值，得到：{posterior!r}")
        if not isinstance(record["ts"], str) or not record["ts"]:
            raise ValueError(f"ts 必须是 ISO8601 字符串，得到：{record['ts']!r}")

    def append(self, record: dict) -> None:
        """追加一条记录（JSONL 一行）；非法记录抛 ValueError 且不落盘。"""
        self.validate_record(record)
        with open(self.path, "a", encoding="utf-8") as f:  # 追加模式，不做全量重写
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def records(self) -> list[dict]:
        """按写入顺序读回全部记录；文件不存在返回空列表。"""
        if not self.path.exists():
            return []
        out: list[dict] = []
        with open(self.path, encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"账本 {self.path} 第 {lineno} 行不是合法 JSON：{e}") from e
        return out


def _node_domain_map(graph: dict) -> dict[str, str]:
    """构建 node_id → domain_id 映射（见模块 docstring 口径 4）。"""
    nodes = {n["id"]: n for n in graph.get("nodes", []) if isinstance(n, dict) and "id" in n}
    mapping: dict[str, str] = {}

    def resolve(nid: str, seen: frozenset[str] = frozenset()) -> str | None:
        if nid in seen:  # 防御：belongs_to 环（合法图不应出现）
            return None
        node = nodes.get(nid)
        if node is None:
            return None
        layer = node.get("layer")
        if layer == "domain":
            return nid
        if layer == "hypothesis":
            return node.get("domain")
        if layer == "action":
            parent = node.get("belongs_to")
            return resolve(parent, seen | {nid}) if isinstance(parent, str) else None
        return None

    for nid in nodes:
        domain = resolve(nid)
        if domain is not None:
            mapping[nid] = domain
    return mapping


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def domain_report(
    records: list[dict],
    graph: dict,
    gain_key: str = DEFAULT_GAIN_KEY,
) -> dict[str, dict]:
    """按领域聚合账本收益与状态计数，给下轮 prior 调整建议（SPEC §5）。

    输出：``{domain_id: {title, confirmed, refuted, uncertain, blocked,
    total_gain, prior_suggestion}}``；图内全部 domain 节点都会出现（未探索
    领域计数为 0、prior_suggestion 为 hold）。

    total_gain 口径【工程默认值】：仅 confirmed 记录的 ``metrics[gain_key]``
    数值求和（uncertain "不计入战果"，03 文档 H3 纪律；refuted/blocked 同）。

    prior_suggestion 规则（SPEC §5）：
    - confirmed≥1 且有正收益 → "up"（03 文档：接线域确认高收益→prior 上调）；
    - refuted≥1 或 uncertain≥1 且无正收益 → "down"（加权域边际→降权）；
    - 其余（含未探索）→ "hold"（特征域未探索→保留）。
    """
    node2domain = _node_domain_map(graph)
    report: dict[str, dict] = {}
    for node in graph.get("nodes", []):
        if isinstance(node, dict) and node.get("layer") == "domain":
            report[node["id"]] = {
                "title": node.get("title", ""),
                "confirmed": 0,
                "refuted": 0,
                "uncertain": 0,
                "blocked": 0,
                "total_gain": 0.0,
                "prior_suggestion": "hold",
            }

    for rec in records:
        if not isinstance(rec, dict):
            continue
        domain_id = node2domain.get(rec.get("node_id"))
        if domain_id is None or domain_id not in report:
            continue  # 无法映射到领域的记录不计入（口径 4）
        bucket = report[domain_id]
        status = rec.get("status")
        if status in ("confirmed", "refuted", "uncertain", "blocked"):
            bucket[status] += 1
        metrics = rec.get("metrics")
        if status == "confirmed" and isinstance(metrics, dict):
            gain = metrics.get(gain_key)
            if _is_number(gain):
                bucket["total_gain"] += float(gain)

    for bucket in report.values():
        has_positive_gain = bucket["total_gain"] > 0
        if bucket["confirmed"] >= 1 and has_positive_gain:
            bucket["prior_suggestion"] = "up"
        elif (bucket["refuted"] >= 1 or bucket["uncertain"] >= 1) and not has_positive_gain:
            bucket["prior_suggestion"] = "down"
        else:
            bucket["prior_suggestion"] = "hold"
    return report
