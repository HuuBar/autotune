# AutoTune
针对Python算法项目的自动优化Agent

## 运行环境
需要Linux环境，且依赖:
- Python >= 3.12
- Docker 尽量使用新版本，版本过低会无法拉取基础镜像

## 使用说明
配置 `config.yaml`:
1. 沙箱环境配置:
  - 设置 `sandbox.docker.name` 镜像名称，必须保证当前环境中没有重名的镜像
  - 按项目需求更改 `sandbox.docker.image` 基础镜像版本
  - 如果项目有pip依赖包，将requirements.txt文件路径写入 `sandbox.docker.requirements_path`
  - 配置完后可运行 `sandbox_test.py` 测试docker镜像是否能运行项目脚本 (修改自定义执行命令`command`)

2. 将待优化的项目放在Linux环境中，且保证已git初始化，并配置 `workspace.root_dir` 路径
3. 将用户指令写入AGENTS.md文件，并配置 `workspace.agents_md` 路径
4. 创建git_diff文件夹用于存放代码修改的diff文件，并配置 `workspace.git_diff_dir` 路径
5. 推荐创建项目目录结构的说明文件，并配置 `workspace.structure_filepath` 路径
6. 如果有自定义的算法优化相关的skills，将目录的绝对路径写入 `skills.planner_skill_path` 列表
7. 配置个人的 `tavily.api_key` 和 `proxy`

运行:
在 `autotune/`目录下执行 `python main.py`

运行结束后:
1. 正常情况下，`plan_log_save_dir` 中保存了所有方案的总结记录，`git_diff_dir` 中保存了所有方案的执行diff文件
2. 运作日志在 `autotune/log/` 中，可使用 `cd trace && python log_viewer.py --log <logfile_path>` 生成可视化运行轨迹网页

## 注意事项
1. 调用黄区内部模型无需代理，但 tavily 需要代理，所以需要在 `cofig.yaml` 中配置 `proxy`。
2. 网络搜索工具使用 tavily，需要自己注册api_key，每月1000次免费调用额度。
4. grep search工具有文件后缀过滤机制，可通过修改 `grep_allowed_suffix` 配置修改搜索范围。
5. skills 路径的配置中推荐放md所在文件夹的绝对路径，不要放SKILL.md文件的绝对路径。
6. **如果发现Manager在任务刚开始跳过Planner直接调度Developer**，建议中断程序重试，正常应该先调度Planner。

## 待演进
1. 当前仅支持项目以 `master` 为基础分支，后续可自定义 base_branch。
2. 数据分析工具 `inspect_dataframe` 当前只支持读取 csv,tsv,xlsx,xls,json,jsonl 格式文件。
