#!/usr/bin/env bash
# M5 端到端一键流水线：编译 → 校验 → 模拟执行 → 回写 → 聚合报告
# 用法：bash scripts/run_e2e.sh [--out e2e_out/] [--live] [更多参数见 --help]
set -euo pipefail

cd "$(dirname "$0")/.."
exec python -m knowledge_pipeline.e2e.run_pipeline "$@"
