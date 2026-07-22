"""
数据资产预检工具（W3-5）

背景事故：数据资产缺陷（重复列名等）只能在训练期爆炸，曾造成约 36 步排雷损耗。
本工具供 Planner 在侦查阶段对数据资产做一次体检，把缺陷暴露时机从训练期前移到规划期。

检查项：
1. 重复列名（CSV 原始表头级别检测，pandas 读取会静默改名 a -> a.1 掩盖问题）；
2. 各列 dtype；
3. 各列缺失率；
4. 标签分布（传入 label_column 时，含各类别样本量与占比）。
"""
import csv
import json
import os
from collections import Counter
from typing import Optional

import pandas as pd
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

from utils.config import global_config
from utils.file_helper import get_secure_path


VALIDATE_DATA_DESC = """数据资产预检工具。在制定方案前对数据集做一次体检，把数据缺陷暴露在规划期而非训练期。

【输出内容】：
1. 重复列名告警（训练期爆炸的高发隐患）；
2. 各列 dtype 一览；
3. 各列缺失率；
4. 标签分布（传入 label_column 时）：各类别样本量与占比，用于判断类别不平衡程度。
【支持格式】：.csv / .parquet"""


class ValidateDataSchema(BaseModel):
    """Input schema for `validate_data` tool"""
    file_path: str = Field(description="数据文件的项目相对路径（.csv / .parquet）")
    label_column: Optional[str] = Field(
        default=None,
        description="标签列名。传入时额外输出标签分布（各类别样本量与占比）；不传入则跳过",
    )


class ValidateDataTools:
    MAX_DTYPE_ROWS = 200       # dtype/缺失率表最大行数，超出截断
    MAX_LABEL_CLASSES = 50     # 标签分布最大类别数，超出截断

    def __init__(self) -> None:
        self.project_root = global_config.get("workspace", {}).get("root_dir", "")

    def _detect_duplicate_columns_csv(self, abs_path: str) -> list[str]:
        """读取 CSV 原始表头，检出重复列名（pandas 会静默改名，必须读原始表头）。"""
        with open(abs_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, [])
        counts = Counter(col for col in header)
        return sorted([col for col, n in counts.items() if n > 1])

    def _validate_data_impl(self, file_path: str, label_column: Optional[str] = None) -> str:
        # 1. 安全路径与格式校验
        try:
            abs_path = get_secure_path(self.project_root, file_path)
        except PermissionError as e:
            return f"❌ 安全错误: {str(e)}"
        if not os.path.isfile(abs_path):
            return f"❌ 错误: 文件不存在 ({file_path})"
        lower = abs_path.lower()
        if not lower.endswith((".csv", ".parquet")):
            return f"❌ 错误: 不支持的格式，仅支持 .csv / .parquet ({file_path})"

        # 2. 重复列名检测（原始表头级别）
        duplicate_cols: list[str] = []
        if lower.endswith(".csv"):
            try:
                duplicate_cols = self._detect_duplicate_columns_csv(abs_path)
            except Exception as e:
                return f"❌ 错误: 无法读取 CSV 表头 ({str(e)})"

        # 3. 加载数据
        try:
            if lower.endswith(".csv"):
                df = pd.read_csv(abs_path)
            else:
                df = pd.read_parquet(abs_path)
        except Exception as e:
            return f"❌ 错误: 数据加载失败 ({str(e)})"

        # 4. 组装报告
        lines = [
            f"数据预检报告: {file_path}",
            f"规模: {df.shape[0]} 行 × {df.shape[1]} 列",
            "=" * 60,
        ]

        # 4.1 重复列名（最高优先级告警）
        if duplicate_cols:
            lines.append(f"🚨 重复列名告警: 检出 {len(duplicate_cols)} 个重复列名 -> {duplicate_cols}")
            lines.append("   （注意：pandas 读取时会将其静默改名为 `列名.1`，训练管线直接取列名可能取错列！）")
        else:
            lines.append("✅ 重复列名: 未检出")

        # 4.2 dtype 一览
        lines.append("-" * 60)
        lines.append("各列 dtype:")
        dtype_items = list(df.dtypes.items())
        for col, dtype in dtype_items[: self.MAX_DTYPE_ROWS]:
            lines.append(f"  {col}: {dtype}")
        if len(dtype_items) > self.MAX_DTYPE_ROWS:
            lines.append(f"  ... (其余 {len(dtype_items) - self.MAX_DTYPE_ROWS} 列已截断)")

        # 4.3 缺失率
        lines.append("-" * 60)
        missing = df.isna().mean()
        missing_nonzero = missing[missing > 0].sort_values(ascending=False)
        if missing_nonzero.empty:
            lines.append("缺失率: 所有列均无缺失 ✅")
        else:
            lines.append(f"缺失率（仅列出有缺失的 {len(missing_nonzero)} 列，降序）:")
            for col, rate in list(missing_nonzero.items())[: self.MAX_DTYPE_ROWS]:
                lines.append(f"  {col}: {rate:.2%} (缺失 {int(df[col].isna().sum())} 行)")

        # 4.4 标签分布
        if label_column:
            lines.append("-" * 60)
            if label_column not in df.columns:
                lines.append(f"❌ 标签列不存在: {label_column}（现有列: {list(df.columns)[:20]}...）")
            else:
                counts = df[label_column].value_counts(dropna=False)
                total = len(df)
                lines.append(f"标签分布（列: {label_column}，共 {total} 样本）:")
                for label, n in list(counts.items())[: self.MAX_LABEL_CLASSES]:
                    lines.append(f"  {label!r}: {n} 样本 ({n / total:.2%})")
                if len(counts) > self.MAX_LABEL_CLASSES:
                    lines.append(f"  ... (其余 {len(counts) - self.MAX_LABEL_CLASSES} 个类别已截断)")

        report = "\n".join(lines)
        max_chars = global_config.get("agent", {}).get("max_chars")
        if max_chars and len(report) > max_chars:
            report = report[:max_chars] + "\n... (报告过长已截断)"
        return report

    def create_validate_data_tool(self):
        return StructuredTool.from_function(
            name="validate_data",
            description=VALIDATE_DATA_DESC,
            func=self._validate_data_impl,
            infer_schema=False,
            args_schema=ValidateDataSchema,
        )
