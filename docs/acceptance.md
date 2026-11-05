# 开发交付与验收记录

依据：用户提供的 PRD / 技术方案 V1.2，及本次补充“通用环境，只开发，不在本机部署任何东西”。文档中的设计指令按需求解释；文档本身不授予部署、使用凭据或上传私有数据的权限。

## 功能与证据映射

| 范围 | 已开发内容 | 可检查证据 |
|---|---|---|
| M0 数据/匹配 | Pydantic契约、Unicode左闭右开、30人工匹配样例、exact优先、最大IoU诊断、文档/父文档/内容隔离 | `tests/test_match.py`、`test_contracts_metrics.py` |
| M1 确定性指标 | micro/macro/weighted F1、逐类及高风险宏/最差召回、风险权重、独立误删/漏删、字符指标、边界IoU、时延/Token成本证据 | `evaluation.py`、DeepEval三个BaseMetric及原生evaluate测试 |
| M2 六阶段观测 | 实际TXT/DOCX/PDF/图片、OCR适配、S1–S5中间产物、保长替换、PDF底层删除、真实导出复开、CER/内容召回/顺序指标 | `test_pipeline.py`、`test_native_integrations.py`、`traces.jsonl` |
| M3 对比/归因 | 双轨HTML/JSON/MD报告、筛选/CSV/字符下钻、分层CI、配对比较、Date误报降幅CI、S2/S3及联合替代实验 | `report-ui.test.mjs`、`test_end_to_end.py`、`attribution/attribution.json` |
| M4 门禁 | L1/L2/L3、硬软阈值、覆盖/缺失/NaN/执行失败拦截、单类短板、策略指纹、逐失败指标样本与复现命令、CI配置 | `test_operations.py`、`verification/acceptance.json` |
| M5 闭环 | 规则前置日期灰区/judge预算/校准、三态人工反馈、双人一致、新版本回流、分层QC计划、Kappa、漂移告警、门禁健康度、审批记录 | `test_governance_dates.py`、`test_feedback_dataset.py`、CLI各入口 |
| 工程能力 | 有界重试、超时/限流、16并发/200分片、文件和版本缓存、断点隔离、留出访问审计、依赖锁定、wheel/sdist及可移交CI/Dockerfile | `test_operations.py`、`test_end_to_end.py`、`uv.lock`、`dist/` |
| 外部接入代码 | 通用检测HTTP、chat-completions、整条SUT HTTP、OCR HTTP/Tesseract、Langfuse 5.x、PostgreSQL、ClickHouse | 协议测试、SDK内存导出、SQLite测试；真实连接状态见下文 |

## 已执行的开发检查

- Python 3.11.11 下完整 `pytest -q`：85 项通过，包括 DeepEval 原生异步 `evaluate()`、Langfuse 原生 OTel 内存导出、12份PDF并发入口同线程回归、DOCX样式/表格保持及PDF底层文本确实移除。
- `npm test`：3 项 DOM 测试通过，覆盖分页/筛选、Unicode字符高亮、脚本内容转义、模态详情、CSV全部筛选结果导出。
- `ruff check .`、`ruff format --check .`、`compileall`、`uv lock --check --offline`：通过。
- wheel/sdist 构建通过；将 wheel 安装到项目内隔离目录，验证包内默认配置和 HTML 模板均可加载。
- 实际源扫描图片和脱敏图片已查看，确认CJK字体可读，PII所在行已覆盖，业务日期和订单号保留。图片使用行级框，覆盖整行的可用性影响已在接入文档说明。

## 全量数据与运行结果

最终数据为 `data/generated/final`，版本 `synthetic-v1-seed42-efcd4c5a7e`：14,000条标准文本、2,000份实际文件，12类实体；8类复杂形态各250份，包括Word单列/双列表格、PDF双栏/混排/跨页、扫描/印章/合成边注。语言覆盖中文、英文、中英混排，包含嵌套实体、称谓粘连、业务日期和订单难负样本、OCR数字混淆。

切分为11,200 train、1,700 dev、1,600 holdout、1,500 gate。金标由数据构造位置独立生成，不读取检测器输出；无真实客户PII，无伪造人工标注员或数据集Kappa。

`reports/final/full/report.json` 记录实际16,000条结果及执行版本指纹；`report.html` 是可直接打开的下钻报告。规则基线标准文本F1=1.0000，复杂文件F1约0.9948，执行失败0。**这些是合成工程测试结果，不能作为真实 Teacher/Student 模型效果或生产性能指标。**

`reports/final/verification/acceptance.json` 记录完整1,500条门禁集三次独立复跑（不使用缓存）及五类实际SUT故障版本。两个轨道的F1极差为0，六阶段记录完整，五类故障均由目标门禁拦截；正常工程对照可PASS，`release_ready=false`。指标/时间以最终JSON为准。

替代实验在188份复杂门禁文件上分别重跑：基线F1=0.99512195，金标OCR/联合替换后F1=1.0000，布局单独替换F1不变。OCR恢复约0.4878个百分点，配对95%区间约[0.2125,0.7919]个百分点。该结论只说明当前合成回放损伤可归因到OCR；布局无变化不能外推“真实复杂排版不存在问题”。

两种规则演示模型在同一门禁集的F1差为0，配对区间包含0，不声称Teacher或Student显著更优。Date难负样本误报均为0，降幅未定义，报告不会伪造86%。真实Token账单没有提供，成本标记为未实测。

## 明确未完成的环境验收

本次没有部署应用、数据库、容器或常驻服务，没有调用真实模型、LLM judge、Tesseract或OCR HTTP服务，没有连接Langfuse/PostgreSQL/ClickHouse。对应代码已实现并进行契约/SDK/无服务验证，但真实连通性、服务版本兼容、配额、账单、GPU性能和真实数据分布，必须在用户指定的已有环境中补验，不能称为本机已部署或生产验收通过。

内置浏览器拒绝 `file://` 协议，因而没有进行真实浏览器截图与视觉回归。没有另起HTTP服务、换浏览器或绕过该策略；已执行纯内存DOM检查。复杂PDF/DOCX的商业级格式保真、真实手写/印章/倾斜OCR、跨页长实体映射仍需真实解析SUT及业务样本验收。文件可复开不等于版式完全无损。

权重、阈值及复核签字仍需业务/合规确认。合成数据和本地审批文件不能代替真实人工一致性、组织身份认证或合规签字。模型训练、蒸馏、正式标注平台UI、生产部署不在两份文档定义的本期开发范围内。
