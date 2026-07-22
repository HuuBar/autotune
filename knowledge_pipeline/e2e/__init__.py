"""M5：端到端一键流水线（编译 → 校验 → 模拟执行 → 回写 → 聚合报告）。

接口契约：docs/knowledge-pipeline/SPEC.md §7。
入口：``python -m knowledge_pipeline.e2e.run_pipeline`` 或 ``scripts/run_e2e.sh``。

注意：本包不在 ``__init__`` 中 eager import ``run_pipeline``（避免
``python -m`` 双重装载告警），请直接从
``knowledge_pipeline.e2e.run_pipeline`` 导入。
"""
