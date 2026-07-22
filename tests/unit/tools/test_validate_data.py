"""W3-5：数据预检工具 validate_data。"""
import pytest

from tests.conftest import global_config
from tools.code.validate_data import ValidateDataTools
from agent.factory import PlannerAgentBuilder


@pytest.fixture()
def tools(tmp_path):
    global_config["workspace"]["root_dir"] = str(tmp_path)
    return ValidateDataTools(), tmp_path


class TestDuplicateColumns:
    def test_duplicate_columns_detected(self, tools):
        t, tmp = tools
        # 真实事故原型：CSV 表头含重复列名（pandas 读取会静默改名为 label.1）
        (tmp / "train.csv").write_text(
            "feat_a,label,feat_a,label\n"
            "1,0,2,1\n"
            "3,1,4,0\n",
            encoding="utf-8",
        )
        res = t._validate_data_impl("train.csv")
        assert "重复列名告警" in res
        assert "'feat_a'" in res and "'label'" in res

    def test_no_duplicate_passes(self, tools):
        t, tmp = tools
        (tmp / "clean.csv").write_text("a,b,c\n1,2,3\n", encoding="utf-8")
        res = t._validate_data_impl("clean.csv")
        assert "重复列名: 未检出" in res


class TestDtypeAndMissing:
    def test_dtype_listed(self, tools):
        t, tmp = tools
        (tmp / "d.csv").write_text("num,text\n1,x\n2,y\n", encoding="utf-8")
        res = t._validate_data_impl("d.csv")
        assert "num: int64" in res
        assert "text: object" in res

    def test_missing_rate_reported(self, tools):
        t, tmp = tools
        (tmp / "m.csv").write_text("a,b\n1,\n2,x\n3,\n", encoding="utf-8")
        res = t._validate_data_impl("m.csv")
        assert "b: 66.67%" in res  # 3 行中缺失 2 行

    def test_no_missing(self, tools):
        t, tmp = tools
        (tmp / "nm.csv").write_text("a\n1\n2\n", encoding="utf-8")
        res = t._validate_data_impl("nm.csv")
        assert "无缺失" in res


class TestLabelDistribution:
    def test_label_distribution(self, tools):
        t, tmp = tools
        rows = ["label"] + ["0"] * 80 + ["1"] * 20  # 正样本 20%
        (tmp / "labeled.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
        res = t._validate_data_impl("labeled.csv", label_column="label")
        assert "标签分布" in res
        assert "20 样本 (20.00%)" in res
        assert "80 样本 (80.00%)" in res

    def test_missing_label_column(self, tools):
        t, tmp = tools
        (tmp / "l.csv").write_text("a\n1\n", encoding="utf-8")
        res = t._validate_data_impl("l.csv", label_column="nope")
        assert "标签列不存在" in res


class TestSafety:
    def test_missing_file(self, tools):
        t, _ = tools
        assert "文件不存在" in t._validate_data_impl("ghost.csv")

    def test_unsupported_format(self, tools):
        t, tmp = tools
        (tmp / "x.txt").write_text("hi", encoding="utf-8")
        assert "不支持的格式" in t._validate_data_impl("x.txt")

    def test_path_traversal_blocked(self, tools):
        t, _ = tools
        assert "安全" in t._validate_data_impl("../../etc/passwd")


class TestPlannerRegistration:
    def test_validate_data_in_planner_tools(self):
        builder = PlannerAgentBuilder(llm_name="dsv4", store=None, checkpointer=None)
        names = [t.name for t in builder.get_tools()]
        assert "validate_data" in names
