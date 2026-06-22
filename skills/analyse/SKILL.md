---
name: analyse
description: metro_arrive项目 优化策略指引。
---

# Analyse Skill

## 使用场景
用于 metro_arrive(地铁到站预测) 项目的cluster集群效果/覆盖率优化。

## 前置分析
- 在开始制定方案前，必须完全理解 `cluster/clusters.py` 中的集群构建代码，和 `cluster/run.py` 的测试代码。
- 如果需要读取 `cluster/data/` 目录下的数据文件，注意必须用 `head` 或 `tail` 只查看头部或尾部的少量数据即可。
- 测试结果报告会放在 `cluster/quantify_result/` 目录下，必要时你可以使用 `inspect_dataframe` 工具了解测试数据。

## 优化策略
- 当前cluster的特征有：时间正弦、时间余弦、工作日/周末隔离、星期数onehot。制定方案时你可以尝试修改/增删任意特征的算法或权重。
- 还可以尝试修改cluster的eps参数，或者增加兜底策略逻辑。

## 限制禁令
- 禁止修改 `cluster/data/` 目录下的文件。
- 禁止修改 `cluster/run.py` 测试代码。
- 尽量不要修改 `cluster/clusters.py` 中负责评测的 `quantify` 和 `evaluate_coverage` 函数代码。
