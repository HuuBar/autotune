import os
import shutil
import tempfile
import subprocess

from datetime import datetime
from langgraph.checkpoint.memory import InMemorySaver
# from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.memory import InMemoryStore
# from langgraph.store.postgres import PostgresStore

from agent.factory import (
    ManagerAgentBuilder,
    PlannerAgentBuilder,
    DeveloperAgentBuilder,
    DebuggerAgentBuilder,
)
from utils.config import global_config
from utils.logger import logger


class AutoTuneAgent:
    def __init__(
        self
    ) -> None:
        self.store = InMemoryStore()
        self.checkpointer = InMemorySaver()

    def _check_config(self):
        if not global_config.get("workspace", {}).get("root_dir"):
            raise RuntimeError(f"❌ 致命错误：config 中必须指定 workspace/root_dir")
        if not global_config.get("workspace", {}).get("agents_md"):
            raise RuntimeError(f"❌ 致命错误：config 中必须指定 workspace/agents_md")
        if not global_config.get("workspace", {}).get("git_diff_dir"):
            raise RuntimeError(f"❌ 致命错误：config 中必须指定 workspace/git_diff_dir")
        if not global_config.get("agent", {}).get("max_chars"):
            raise RuntimeError(f"❌ 致命错误：config 中必须设置 agent/max_chars")

    def _export_plan_log(self, save_dir: str):
        """
        从 Store 中提取 plan_log.md 并保存到指定目录
        """
        target_key = "/plan_log.md"
        namespace = ("filesystem",)
        try:
            item = self.store.get(namespace, target_key)
            if not item:
                logger.info(f"未找到 {target_key}，本次运行没有生成长期记忆。")
                return

            raw_data = item.value
            if isinstance(raw_data, dict):
                content_data = raw_data.get("content", raw_data)
            else:
                content_data = raw_data

            if isinstance(content_data, list):
                file_content = "\n".join(str(line) for line in content_data)
            elif isinstance(content_data, str):
                file_content = content_data
            else:
                file_content = str(content_data)

            os.makedirs(save_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%m%d_%H%M")
            export_file_name = f"plan_log_{timestamp}.md"
            save_path = os.path.join(save_dir, export_file_name)
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(file_content)
            logger.info(f"✅ 成功导出 plan_log.md: 已保存至 {save_path}")
        except Exception as e:
            logger.error(f"❌ 导出 plan_log 时发生异常: {str(e)}")

    def warmup_docker_env(self):
        """
        进程启动时调用：docker 镜像构建。
        """
        logger.info("⏳ [沙箱基建] 正在校验并预热 Docker 运行环境 ...")

        sandbox_cfg = global_config.get("sandbox", {}).get("docker", {})
        image_id = sandbox_cfg.get("name", "autotune-env")
        base_image = sandbox_cfg.get("image", "python:3.12-slim")
        req_path = sandbox_cfg.get("requirements_path", "")
        pip_mirror = sandbox_cfg.get("pip_mirror", "")

        # 1. 测试 Docker
        try:
            subprocess.run(["docker", "info"], check=True, capture_output=True, timeout=5)
        except Exception:
            raise RuntimeError("❌ 致命错误：Docker 未启动或未安装，沙箱初始化失败！")
        # 检查目标镜像是否已存在
        image_name = f"{image_id}:latest"
        try:
            res = subprocess.run(["docker", "images", "-q", image_name], capture_output=True, text=True, check=True)
            if res.stdout.strip():
                logger.info(f"⚡ [沙箱基建] 检测到本地已存在镜像 {image_name}，跳过构建流程！")
                logger.info(f"提示：若您更新了 requirements.txt，请手动在终端执行 `docker rmi {image_name}` 以触发重建。")
                return
        except subprocess.CalledProcessError:
            pass
        # 2. 无 requirements 时，直接基础镜像打Tag
        if not req_path or not os.path.isfile(req_path):
            logger.warning(f"⚠️ 未检测到 requirements.txt，准备使用基础镜像 {base_image} ...")
            try:
                logger.info(f"⬇️ 正在拉取基础镜像 {base_image} (如果本地已有将跳过) ...")
                subprocess.run(["docker", "pull", base_image], check=True)
                subprocess.run(["docker", "tag", base_image, f"{image_id}:latest"], check=True)
                logger.info("✅ [沙箱基建] 基础镜像已就绪！")
                return
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"❌ 致命错误：无法拉取基础镜像 {base_image}，请检查网络或 Docker 配置。\n{e}")
        logger.info(f"检测到依赖文件 {req_path}，开始构建专属运行镜像 {image_id}...")
        # 3. 构建动态 Dockerfile
        with tempfile.TemporaryDirectory() as temp_build_ctx:
            # 将目标依赖文件拷贝到临时目录中
            target_req_name = "requirements.txt"
            temp_req_path = os.path.join(temp_build_ctx, target_req_name)
            shutil.copy(req_path, temp_req_path)
            # 组装 Dockerfile
            dockerfile_content = f"""
            FROM {base_image}
            COPY {target_req_name} /tmp/requirements.txt
            """
            # 处理 pip 镜像
            if pip_mirror:
                dockerfile_content += f"RUN pip config set global.index-url {pip_mirror}\n"
                if "://" in pip_mirror:
                    domain = pip_mirror.split("://")[-1].split("/")[0]
                    dockerfile_content += f"RUN pip config set global.trusted-host {domain}\n"
            dockerfile_content += "RUN pip install --no-cache-dir -r /tmp/requirements.txt"
            # 执行构建 (会覆盖已存在的同名镜像)
            try:
                subprocess.run(
                    ["docker", "build", "-t", f"{image_id}:latest", "-f-", "."],
                    input=dockerfile_content,
                    text=True,
                    check=True,
                    cwd=temp_build_ctx
                )
                logger.info("✅ [沙箱基建] 镜像构建成功！")
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"❌ 致命错误：镜像构建失败。\n构建日志: {e.stderr}")

    def run(self):
        self._check_config()
        self.warmup_docker_env()

        llm = "dsv4"

        manager_builder = ManagerAgentBuilder(llm_name=llm, store=self.store, checkpointer=self.checkpointer)
        planner_builder = PlannerAgentBuilder(llm_name=llm, store=self.store, checkpointer=self.checkpointer)
        developer_builder = DeveloperAgentBuilder(llm_name=llm, store=self.store, checkpointer=self.checkpointer)
        debugger_builder = DebuggerAgentBuilder(llm_name=llm, store=self.store, checkpointer=self.checkpointer)

        planner_subagent = planner_builder.to_subagent()
        developer_subagent = developer_builder.to_subagent()
        debugger_subagent = debugger_builder.to_subagent()
        manager_agent = manager_builder.build(subagents=[planner_subagent, developer_subagent, debugger_subagent])

        recursion_limit = global_config.get("agent", {}).get("recursion_limit")
        input_md = global_config.get("workspace", {}).get("agents_md")
        with open(input_md, 'r', encoding='utf-8') as f:
            user_prompt = f.read()

        try:
            result = manager_agent.invoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": user_prompt
                        }
                    ]
                },
                config={
                    "configurable": {
                        "thread_id": "thread_id"
                    },
                    "recursion_limit": recursion_limit
                }
            )
            print(result)
        except Exception as e:
            logger.error(f"❌ 任务执行中断:\n{str(e)}")
            raise
        finally:
            plan_log_save_dir = global_config.get("workspace", {}).get("plan_log_save_dir", "./log")
            logger.info("正在保存plan日志...")
            self._export_plan_log(plan_log_save_dir)
