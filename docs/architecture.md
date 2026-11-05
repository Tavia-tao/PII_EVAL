# 架构与依赖方向

系统为无常驻服务的评测应用。业务入口是 CLI 或可调用的 `run_evaluation()`；文件、指标、模型、账本与外部服务通过适配器分离。

| 模块 | 职责 | 不应依赖 |
|---|---|---|
| `schema.py` | 样本、标注、预测、六阶段、Token 用量契约 | 外部模型/数据库 |
| `match.py` | exact/partial/wrong_type 配对及唯一 TP/FP/FN | DeepEval、Langfuse、原始文件 |
| `evaluation.py`、`stats.py` | 分层聚合、风险/误删/字符指标、CER、配对 bootstrap | 外部 judge |
| `metrics/` | DeepEval 包装和日期歧义建议 | 重写匹配算法 |
| `adapter/` | 文件解析/OCR/布局/检测/替换/导出、真实服务接入 | 样本金标（归因 override 除外） |
| `runner.py` | 数据加载、版本快照、并发/限流/重试/分片、缓存/断点、报告 | 自动部署 |
| `gate.py` | 完整性、覆盖和阈值门禁，失败清单 | 放宽阈值以绕过 BLOCK |
| `observability.py` | 本地镜像、Langfuse Dataset/Experiment/六阶段/Score | 向远端上传原文/实体值 |
| `feedback.py`、`dataset.py` | 人工审计、双人确认、新版本回流及泄漏防护 | 原地覆盖留出集 |
| `monitor.py` | 从明确时间范围的抽检快照生成告警、健康检查 | 用无金标预测伪造召回 |
| `storage.py` | PostgreSQL 运营记录、ClickHouse 汇总、SQLite 开发测试 | 把不同数据库互相冒充 |

常规推理数据流为 `Sample.input → S1_parse → S2_ocr → S3_layout → S4_detect → S5_mask → S6_export → Prediction`。模型客户端只接收观察文本与模型/Prompt 版本。评测端再读取 `Sample.gold` 与实际中间产物，计算分数。归因入口明确通过 `Overrides` 替换上游产物，单独、联合重跑下游，不根据总分臆测阶段贡献。

`Span` 坐标为 Python Unicode 码点、左闭右开；JS 报告使用 `Array.from(text)` 按相同口径高亮。观察文本与金标文本不一致时，用字符对齐投影到共同参考坐标，无法投影的预测保留为 FP。对齐不会纠正被测模型的输出，也不能代表 OCR 语义恢复正确。

每个样本保存一份 S1–S5 产物 JSON，六阶段记录保存引用与耗时，S6 保存真实脱敏导出。读取与替换统一使用保长字符，嵌套实体按区间并集覆盖。PDF 使用真实 redaction 删除底层文本，DOCX 逐 run 替换并保留现有样式；遇到无法安全处理的超链接/域文本会失败，不静默生成未脱敏文件。图片按 OCR bbox 覆盖，字符级 bbox 不可用时会覆盖整行，可用性影响需单独验收。

PDF 解析和回填保持在同一评测线程，不进入通用线程池；其他文件 I/O 和远程模型仍可异步并发。依据 [PyMuPDF 官方并发说明](https://pymupdf.readthedocs.io/en/latest/recipes-multiprocessing.html)，需要扩展 PDF 吞吐时应使用隔离进程，各进程独立打开文档，不能直接把 PyMuPDF 操作放入多线程。

状态分为数据清单、预测缓存、运行断点、复核审计、可选数据库和 Langfuse 实验，不依赖跨聊天记忆。缓存和断点均绑定版本；失败预测不缓存；缺失/损坏导出会使缓存失效。外部模型/OCR 的地址、密钥名称、并发、超时、预算、风险权重、门禁策略等均由配置/环境提供。
