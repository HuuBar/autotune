# 自动化 ML 优化系统 · 战术知识卡 v1（P0 实验产出）

> **这是什么**：不是调研报告，而是给自动优化 agent 用的原子化知识单元库。
> **筛选规则**（H1 的核心判据）：只收录三类信息——**[决策]** 会改变某个决策的信息、**[动作]** 能直接转成执行动作的信息、**[边界]** 标注做法何时失效的信息。概念综述、营销内容、与"agent 驱动 ML 优化"无关的细节一律弃用。
> **卡片格式**：情境 / 做法 / 依据（含来源实测）/ 边界 / 验证方法 / 对我们的落点。
> **使用方式**：编排层（P1）消费全集做假设排列；运行时 agent 按当前任务检索 top-K 注入，不全量灌入。

---

## A 类 · 搜索与探索策略（对应 P1/P2）

### A1 单原子变更原则 [动作]
**情境**：每次实验迭代。
**做法**：一步只改一件事（一个超参、一个特征、一处修复），改完立即评估。
**依据**：AIDE 的 Improve 算子强制"single, atomic improvement"[^61^]；autoresearch 的 program.md 规定"Make ONE atomic change to train.py"[^93^]。一次改多处则指标变化无法归因。
**边界**：大规模重构（如换框架）天然不可原子化——此时应拆成"可独立验证的最小迁移序列"。
**验证**：检查本轮 diff 是否只服务于一个意图。
**落点**：Developer 的 edit 纪律；也是 todo.md 每个 Step 的粒度标准。

### A2 先广度后深度：草稿-改进-调试三算子 [决策]
**情境**：进入一个新优化问题。
**做法**：先产出 N 个独立"根草稿"（不同方向的基线方案），再贪心选择当前最优节点做改进；节点出错时以一定概率转调试而非放弃。
**依据**：AIDE 的 Draft/Improve/Debug 三算子 + greedy 选优 + debug_prob，MLE-Bench 上奖牌数为最强线性 agent 的 4 倍 [^61^][^76^]。
**边界**：评测昂贵时 N 要小；草稿方向应由假设空间（P1）给出而非随机。
**验证**：前几轮实验是否覆盖了 ≥2 个不同机制方向。
**落点**：直接对应我们的"假设罗列→选优验证"流程；调试概率对应 Developer 排障与推进的配比。

### A3 固定预算可比性 [动作][边界]
**情境**：多方案需要公平比较。
**做法**：给每次实验固定 wall-clock 预算（如 5 分钟），用时间而非 epoch 做归一化。
**依据**：autoresearch 借此让大小模型同台比较，Shopify CEO 跑出 0.8B 超过手调 1.6B 的结果 [^94^]。
**边界**：会引入"吞吐黑客"——在预算内跑更多 step 的方案占便宜，但未必是长训最优（见 B7）。
**验证**：比较结论是否在同一预算口径下得出。
**落点**：我们比较不同 Plan 的训练结果时，必须先统一预算口径再比指标。

### A4 小样本先行，全量殿后 [动作]
**情境**：任何代码改动后的验证。
**做法**：先在采样小数据集上迭代调试到能跑通，再上全量数据评估。
**依据**：RD-Agent 的 Developer 分两阶段——"iteratively debugging on a sampled dataset"后再全量运行，官方明确说这是效率关键 [^83^]。
**边界**：小样本要保留类别分布等关键统计特征，否则调通过的东西全量仍炸。
**验证**：小样本上的通过率高不代表全量通过，但能把 80% 的语法/类型/路径错误拦在 30 秒内。
**落点**：就是我们此前"验证阶梯 L1 冒烟档"的业界实证——v1 的 JSON 事故本可在此层拦截。

### A5 多轨迹并行 + 失败史共享 [决策]
**情境**：同一目标存在多条互斥路线。
**做法**：并行多条探索轨迹（可用不同 prompt/模型/工具配置制造多样性）；后启动的轨迹**先读先启动轨迹的探索史与失败案例**，避免重复无效方向；末期做轨迹融合（A 轨迹的特征工程 + B 轨迹的模型）。
**依据**：RD-Agent 的 multi-trace + 信息交换协议 + fusion，MLE-Bench 上显著超 AIDE [^83^]。
**边界**：无协调的并行会浪费在重复探索上——交换协议是必需品不是可选项。
**验证**：各轨迹的方向重合度；融合方案是否优于任一单轨迹。
**落点**：对应 C-05 中"真实验未知"类假设的并行化；信息交换协议可直接实现为共享的假说账本。

### A6 探索的节奏模式 [决策]
**情境**：长程优化中决定"继续挖还是换方向"。
**做法**：允许"局部细化→方向转移→带新知识回访旧假设"三种节奏交替。
**依据**：RD-Agent(Q) 实测发现三种有效模式：在一个概念簇内连续细化后整体转向；后期回访早期有潜力假设并精化；最终 SOTA 解集 8/36 次试验入选、横跨 6 个簇中的 5 个——**最优解来自多方向覆盖而非单点深挖** [^80^]。
**边界**：频繁无耐心跳转会沦为随机游走——“穷尽一个概念”需要停止判据（见 C3）。
**验证**：复盘已探索方向的簇分布是否过于集中。
**落点**：P1 编排器调度假设的顺序策略。

## B 类 · 评估与验证设计（对应 P3，本批卡中密度最高的一类）

### B1 防作弊指标设计 [决策]
**情境**：定义优化的目标指标时。
**做法**：枚举 agent 有权改动的所有变量，逐一检查"不动真本事能否提分"；对能提分的变量，要么禁止改、要么把指标设计成对其免疫。
**依据**：autoresearch 用 val_bpb（按字节归一）而非 val_loss——若用 token 级 loss，agent 改 vocab_size 就能"提分"而不改进模型 [^94^][^67^]。
**边界**：指标免疫性检查无法穷尽，需配合 B2。
**验证**：做一次"作弊推演"：假设 agent 纯粹想刷分，最省力的路径是什么？
**落点**：我们的"王者荣耀 F1 + 视频类不劣化"就是防单点作弊的约束设计；阈值过拟合验证集则是没堵住的洞（见 B7）。

### B2 信任边界：评估侧文件不可改 [动作]
**情境**：agent 可写代码的项目结构。
**做法**：数据准备、评估函数、提交格式划入"不可修改"区，并在工具层强制（不是提示词层）。
**依据**：autoresearch 把 prepare.py 划为禁区——"若数据处理也可改，agent 可能用'让 eval set 变小'来作弊"[^67^][^105^]。
**边界**：禁区内的 bug agent 修不了，需要人类通道。
**验证**：git 层面检查禁区文件的 diff 恒为空。
**落点**：autotune 的 evaluate/train 入口与 result_stats 生成脚本应划入 Developer 不可写区。

### B3 机器可读结果契约 [动作]
**情境**：实验结果的记录与传递。
**做法**：每次实验产出结构化记录（commit、指标、状态、假设描述），追加进账本文件；下游角色只读账本，不听口述。
**依据**：autoresearch 的 results.tsv（commit/val_bpb/status/description）[^93^]；AIDE 的 Journal 每节点存 plan/code/term_out/metric/is_buggy [^61^]。
**边界**：账本只记"发生了什么"，"为什么"需要另配分析层。
**验证**：任一指标都能溯源到具体 commit 与原始日志。
**落点**：就是我们要建的 experiments/hypotheses 账本；直接回应 v1 的"Manager 无法核验指标"问题。

### B4 keep/discard 决策要过统计关 [决策]
**情境**：判定"这次改动有没有提升"。
**做法**：对噪声大的小预算实验，keep 决策前做重复运行或不确定性建模，而非单跑一次定胜负。
**依据**：autoresearch 社区复盘明确指出"keep 决策可能部分是噪声驱动"，并出现"某些想法跨 session 无法复现" [^101^]；社区工具 autojudge 专门提供"statistical keep/discard verdicts" [^106^]。
**边界**：评估昂贵时至少对"边际提升"（<噪声带宽）做复跑。
**验证**：连续两次同配置运行的指标差（噪声带宽）是否远小于 claimed 提升。
**落点**：F1 提升 0.005 级别的"改进"不能直接入 plan_log 当战果。

### B5 评估外置化 [决策]
**情境**：agent 长期自评估时。
**做法**：评估流程与 agent 执行环境分离——一致、固定、对 agent 不可见其内部细节。
**依据**：AIRA_2 的核心归因：长程性能退化"不是真过拟合，而是噪声评估或被 agent 污染的评估流程"所致，解法是 externalized, consistent, hidden evaluation [^77^]。
**边界**：外置评估也要防 B7 的代理错位。
**验证**：评估代码与数据是否位于 agent 的写权限之外（同 B2）。
**落点**：与 A1 核验工具合并设计。

### B6 选择时计入过拟合风险 [决策]
**情境**：从多个候选解中选最终方案。
**做法**：选择函数不只看验证指标，还看训练/验证曲线形态诊断过拟合风险。
**依据**：RD-Agent 轨迹融合后的最终选择用 composite scoring："validation performance, solution robustness, and overfitting risk, as derived from the score curves and model diagnostics" [^83^]。
**边界**：曲线诊断规则需要按任务类型校准。
**验证**：所选方案在 held-out 集上的衰减幅度。
**落点**：Plan 间选最优基线时加入曲线诊断，而非只看 result_stats 单点。

### B7 代理评估与真实目标的错位 [边界]
**情境**：一切"先便宜评估再推广"的场景。
**做法**：定期用全量/长程评估校准代理指标；对代理指标上的胜出者保持"假设"态度而非"结论"。
**依据**：5 分钟预算强调早期训练动态，"settings that win early may lose later" [^101^]；本批次最典型案例正是我们自己的阈值 0.90——小验证集搜出的顶格阈值。
**验证**：代理指标胜者在大预算下的复测排名保持率。
**落点**：C3 评估科学的理论基础之一；敏感性曲线/交叉验证的必要性。

## C 类 · 失败模式与防御（全是实测坑）

### C1 API/库幻觉 [决策]
**情境**：agent 写 ML 代码调用库函数。
**做法**：给 agent 锚定外部库知识（文档片段、可用 API 清单），不依赖其参数化记忆。
**依据**：MLZero 的错误分析：AIDE 的首要失败模式是 API 幻觉，占 28.2%——"underscores the necessity for external knowledge of ML libraries rather than exclusive reliance on parametric knowledge" [^63^]。
**验证**：统计执行失败中"函数不存在/签名错误"的占比。
**落点**：知识流水线（C2 知识供给）要包含项目所用库的 API 卡，不只是方法论。

### C2 浅编辑/幽灵集成 [动作]
**情境**：agent 在大型存量代码库添加组件。
**做法**：每次新增函数/类后，验证它真的被调用链接入（grep 调用点、跑通后确认行为变化）。
**依据**：FML-bench 实测：AIDE "generated new classes or components that were never integrated into the actual execution pipeline, resulting in no functional improvement" [^68^]。
**验证**：改动前后跑同一入口，确认新代码路径被执行（覆盖率或日志埋点）。
**落点**：Developer 验收清单加一条"集成检查"——v1 本轮 19 处编辑都接入了，但框架层没有强制。

### C3 早停与空转的两难 [决策]
**情境**：长循环中决定继续还是停。
**做法**：停止权不放给模型自由裁量；用规则信号（连续 k 轮信息增益低于阈值则停，k≥2 防单轮噪声触发）。
**依据**：FML-bench 记录 Claude Code"频繁自行提前终止实验" [^68^]；语义早停研究实测"loop earns its keep early and idles late"，用 entropy/信息增益信号 + k=2 patience 可省 38% token 且质量无损 [^85^]。
**边界**：信号阈值按任务调；熔断（异常死循环）与功成身退（无增益）是两套规则。
**落点**：autotune 的熔断机制之外，补"无增益早停"规则——防止 Plan 后期空转。

### C4 错误假设固化与确认偏误 [边界]
**情境**：长轨迹中 agent 基于早期推断行动。
**做法**：定期强制"假设复查"——把当前依赖的关键假设列出并要求逐条拿出证据。
**依据**：长程失败归因研究：agent 把"先验当普适事实"，且"intermediate outputs are interpreted as validation rather than signals to re-check premises"，轨迹越长越不回溯 [^87^]。
**落点**：这正是我们步骤 147（发现阈值异常却放线）的学术命名；Developer 每 N 步插入一次假设复查点。

### C5 环境污染 [动作]
**情境**：agent 可在执行环境装包/写文件。
**做法**：容器隔离每次实验环境；长任务做 checkpoint；失败可回滚到干净态。
**依据**：AIRA-dojo 基础设施教训："agents corrupting Python environments e.g., via a pip install"→容器化；硬/软失败常见→checkpointing [^79^]。
**落点**：autotune 的 reset_sandbox 只删容器不清理挂载目录——outputs/ 残留需纳入隔离设计。

### C6 记忆回授污染 [边界]
**情境**：agent 产出的文档回流为知识库内容。
**做法**：知识库写入必须带 provenance（来源、是否已验证）；未验证的 agent 自产物标记为"待验"。
**依据**：agent 评估研究："agent-generated data fed back into retrieval…entrenches its own errors" [^65^]。
**落点**：技能库/经验库写入时强制来源字段；plan_log 的"经验"区与"已验证事实"区分。

## D 类 · 知识与记忆工程（对应 P0/P4）

### D1 角色按反馈类型分工，模型按角色分配 [决策]
**情境**：多 agent 系统的角色设计。
**做法**：规划者只消费性能反馈、执行者只消费错误反馈；两类角色可用不同模型（强推理模型做规划、强指令遵循模型做执行）。
**依据**：RD-Agent 的 Researcher/Developer 按反馈类型分工；混合后端实测 o3 做研究 + GPT-4.1 做开发"match or exceed strongest baseline"且更省 [^83^]。
**落点**：我们的 Planner/Developer 分工同构；temperature/模型可按角色差异化（v1 是全员 0.7）。

### D2 实验账本的三元组结构 [动作]
**情境**：记录每次探索。
**做法**：每条记录 = 假设（自然语言）+ 实现（代码/commit）+ 反馈（指标+错误），缺一不可。
**依据**：AIDE Journal 节点结构 [^61^]；RD-Agent 的 knowledge forest 以"historical hypotheses + corresponding feedback"驱动新假说生成 G(H,F) [^80^]。
**落点**：假说账本的 schema 基准。

### D3 技能/知识的最小高信号集 [动作]
**情境**：给 agent 写 skill 或知识卡。
**做法**：元信息（名称+描述）常载，正文按需加载；正文只放当前任务最相关指令；2-3 个典型示例优于穷举边界情况。
**依据**：Anthropic 渐进式披露实践："smallest possible set of high-signal tokens"；反例是"将完整 API 文档塞入 SKILL.md" [^91^]。
**落点**：本知识卡文件的使用方式——运行时只注入 top-K，不是全量。

### D4 经验检索而非经验背诵 [动作]
**情境**：agent 需要历史经验时。
**做法**：经验库按描述做语义索引，任务来临时检索 top-K 注入；新经验在验证成功后才入库。
**依据**：Voyager 技能库（成功才存、语义检索 top-5、3.3× 探索效率）[^47^]；检索式预取在工具场景有 49% token 削减 + 3.2× 准确率的实测 [^95^]。
**落点**：C2 知识流水线的存取机制基准。

## E 类 · 编排与接口（对应 P1/P2）

### E1 收窄可变范围 [动作][边界]
**情境**：定义 agent 的行动空间。
**做法**：尽量限定可修改的文件/参数集合，让每个 diff 可审、每个失败被隔离。
**依据**：autoresearch 只放开 train.py——"narrows agent action space, prevents scope creep" [^93^]。
**边界**：真实项目改动天然跨文件——FML-bench 指出 AIDE 的单文件限制使其在真实研究代码库上不足 [^68^]。所以收窄的对象应是"每次实验的意图范围"，而非物理文件数。
**落点**：todo.md 每个 Step 声明"本步只许动哪些文件"并工具层校验。

### E2 指令文件即策略接口 [动作]
**情境**：人要调整 agent 行为。
**做法**：策略写在独立的指令文件（program.md 式），人改文件不改代码；文件保持"可迭代的简陋"，在跑的过程中逐步补策略。
**依据**：autoresearch 的 program.md 被作者称为"super lightweight skill"，人类通过改它调整研究方向 [^105^]；社区甚至发展出"行为式 program.md"（概念+反应规则的写法）[^90^]。
**落点**：我们的 todo.md/plan_log/系统提示应该进一步分化为"人写的策略层"与"系统生成的事实层"。

### E3 无人值守循环的两条铁律 [决策]
**情境**：循环过夜跑。
**做法**：①永不暂停等人（NEVER STOP）；②但每个 keep/discard 决策都有硬性规则兜底，不靠模型临场发挥。
**依据**：autoresearch program.md 的"never pause to ask the human" [^93^] 与 keep/revert 规则化并存。
**落点**：与 C3 配套：自主性与规则约束不是对立面，是一体两面。

## F 类 · 基础设施（对应 P2 支撑）

### F1 节奏预算 [动作]
**情境**：规划实验吞吐量。
**做法**：先测单轮实验耗时，换算"每小时实验数"，据此排并行度与时间盒。
**依据**：autoresearch 实测 5 分钟/轮 ≈ 12 轮/小时 ≈ 过夜百轮 [^97^]；其探索节奏"早期广撒、后期收窄" [^97^]。
**落点**：v1 单 Plan 26 分钟——意味着过夜可跑的 Plan 数很少，假设排序质量比数量更关键（呼应 C-04）。

### F2 git 即审计轨迹 [动作]
**情境**：实验历史管理。
**做法**：每次 keep 一个原子 commit，分支即血统；失败 revert 不进主干。
**依据**：autoresearch 的 ratchet 机制："codebase can only move forward" [^97^]。
**落点**：autotune 已具备（本轮 commit + diff-export 规范），列为已验证资产。

### F3 限流与重试是常态需求 [边界]
**情境**：依赖外部 LLM 服务。
**做法**：按限流设计并发上限；重试+退避；关键状态落盘可续跑。
**依据**：AIRA-dojo：外部 LLM 服务会变慢导致超时，自托管是可靠 scaling 的前提；缓存与指数退避是标配 [^79^][^85^]。
**落点**：v1 的 ModelRetryMiddleware 已有雏形，补状态持久化（对应 B 类工程项）。

---

## 与理想流程的映射总表

| 阶段 | 覆盖卡片 |
|---|---|
| P0 知识备料 | D3 D4 C6 |
| P1 信息组织 | A2 A5 A6 D1 D2 E2 |
| P2 编排执行 | A1 A3 A4 E1 E3 F1 F2 F3 C2 C5 |
| P3 验证回写 | B1 B2 B3 B4 B5 B6 B7 C3 C4 |
| P4 沉淀增殖 | D2 D4 C6 F2 |

## H1 自评与人工评审指引

**自评（供你对照）**：本批 24 张卡全部过三档筛选；其中[决策]类 12 张、[动作]类 13 张、[边界]类 8 张（有重叠）。**有意弃用的内容**：各系统的架构图细节、benchmark 排名数字（除用作依据的 4×/28.2% 等决策相关数字）、纯理论方法（如 MAD 辩论、ToT）——按规则它们不改变我们的动作。

**请你评审的三个点**：
1. **档位误判**：有没有哪张卡你觉得"这信息没用"（我筛错了），或你期待有的信息类型缺了（我漏了）？
2. **实操密度**：每张卡的"做法"是否具体到能转动作？哪几张还停留在"正确的废话"？
3. **格式**：这个卡片形态能否直接进入你的细粒度编排处理，还是需要调整 schema（比如加前置条件/成本字段）？

## 参考来源

[^47^]: https://github.com/Shoepon/Voyager-Ollama
[^61^]: https://thedocumentation.org/aideml/concepts/agentic_tree_search/
[^63^]: https://assets.amazon.science/78/c9/c0a204f24927922458f00df35f5c/scipub-approval152129-37266059-auto2ml-a-multiagent-system-for-automated-endtoend-machine-learning-solutions.pdf
[^64^]: https://www.emergentmind.com/topics/mle-bench
[^65^]: https://mentiora.ai/blog/agent-test-score/
[^67^]: https://github.com/ktchuang/TAICA_AIASE2026/blob/main/W7.md
[^68^]: https://arxiv.org/html/2510.10472v1
[^76^]: https://github.com/wecoai/aideml
[^77^]: https://www.emergentmind.com/papers/2603.26499
[^79^]: https://arxiv.org/html/2507.02554v1
[^80^]: https://saulius.io/blog/automated-quant-research-ai-agents-rd-agent
[^83^]: https://arxiv.org/html/2505.14738v1
[^85^]: https://arxiv.org/pdf/2606.27009
[^87^]: https://arxiv.org/html/2604.11978v1
[^90^]: https://github.com/BarishNamazov/interpretable-autoresearch
[^91^]: https://github.com/wenshao/codeagents/blob/main/docs/comparison/skill-system-deep-dive.md
[^93^]: https://github.com/sergiocoding96/hermes-multi-agent/blob/main/PROJECT-STATE-2026-04-05.md
[^94^]: https://pub.towardsai.net/karpathy-left-his-gpu-running-overnight-the-agent-found-a-bug-everyone-missed-for-months-eae7e351104d
[^95^]: https://github.com/garrytan/gbrain/blob/master/skills/functional-area-resolver/SKILL.md
[^97^]: https://www.datacamp.com/tutorial/guide-to-autoresearch
[^101^]: https://kingy.ai/ai/autoresearch-karpathys-minimal-agent-loop-for-autonomous-llm-experimentation/
[^105^]: https://github.com/karpathy/autoresearch
[^106^]: https://github.com/yibie/awesome-autoresearch
