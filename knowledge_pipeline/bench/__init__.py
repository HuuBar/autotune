"""M6：量化评测台 —— 事故剧本库 + 有/无编排 A/B 对照 + 预登记指标。

语义来源：SPEC §8、03 文档第四部分（L1 离线 dry-run A/B；剧本对双方完全
一致；过程指标为主判据；每剧本独立计分再聚合）。
"""

from .baseline_agent import (
    CARDS,
    EVENT_VOCAB,
    INITIAL_PROBE,
    BaselineAgent,
    BaselineResult,
    run_baseline,
)
from .metrics import METRIC_NAMES, aggregate, canonical_events, compute_metrics
from .runner import run_bench, run_scenario
from .scripts import (
    SCENARIO_IDS,
    Scenario,
    ScriptError,
    default_scripts_dir,
    load_scenario,
    load_scripts,
)

__all__ = [
    "SCENARIO_IDS",
    "Scenario",
    "ScriptError",
    "default_scripts_dir",
    "load_scenario",
    "load_scripts",
    "CARDS",
    "EVENT_VOCAB",
    "INITIAL_PROBE",
    "BaselineAgent",
    "BaselineResult",
    "run_baseline",
    "METRIC_NAMES",
    "aggregate",
    "canonical_events",
    "compute_metrics",
    "run_bench",
    "run_scenario",
]
