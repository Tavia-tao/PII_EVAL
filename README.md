# PII 脱敏模型评测系统

按照用户提供的《PII脱敏模型评测_产品需求文档》《PII脱敏模型评测_技术方案文档》V1.2 开发。覆盖确定性 span 指标、六阶段流水线、双轨报告、配对统计比较、三级门禁、人工复核、Bad Case 回流及漂移监控。**不包含模型训练/蒸馏，也不伪造真实模型评测成绩。**

本次只开发和测试这个项目。没有部署数据库、应用、容器或常驻服务，没有连接外部模型或 Langfuse 实例。`.venv`、`node_modules`、生成数据与测试输出都位于项目目录，属于开发依赖和文件。

Git 仓库包含源码、配置、测试与文档；生成数据、评测报告、图片输出、本地运行状态和依赖目录不纳入版本控制。下文的最终数据与报告路径属于本地交付产物，新克隆项目后请按“直接运行”中的命令生成数据并运行评测。

## 交付入口

- [功能验收及验证范围](docs/acceptance.md)
- [指标口径与文档冲突决策](docs/decisions.md)
- [模块职责与数据流](docs/architecture.md)
- [接入现有模型、OCR、Langfuse、数据库](docs/integrations.md)
- [运维、数据版本、复核回流与故障处理](docs/operations.md)
- 最终数据：`data/generated/final/manifest.json`，14,000 条文本、2,000 份实际文件。
- 全量报告：`reports/final/full/report.html`，双击即可查看；无需启动服务。
- 工程验收：`reports/final/verification/acceptance.json`。

## 环境与依赖

| 组件 | 开发验证版本 | 用途 |
|---|---|---|
| Python | 3.11.11 | 与技术文档 Python 3.11 一致 |
| DeepEval | 4.3.3 | 原生 BaseMetric、自定义确定性指标、歧义 GEval 适配 |
| Langfuse Python SDK | 5.2.0 | OTel 六阶段链路、版本化 Dataset、Experiment、Score、复核队列 |
| Pydantic | 2.14.3 | 样本及预测契约 |
| PyMuPDF / python-docx | 1.28.2 / 1.2.0 | 实际文件解析、脱敏、导出复开 |
| SQLAlchemy / psycopg | 2.1.8 / 3.3.8 | PostgreSQL 运营记录；SQLite 用于无服务测试 |
| clickhouse-connect | 0.15.1 | ClickHouse 分析结果写入 |
| Node / jsdom | Node ≥24 / 27.4.0 | 仅报告 DOM 测试，不是运行时前端依赖 |

完整依赖锁定在 `uv.lock` 和 `package-lock.json`。不需要 GPU 或 API Key 就能完成合成工程验收。若在其他机器重新安装，使用项目独立环境：

```powershell
uv sync --frozen --python 3.11
npm ci --ignore-scripts
uv run pytest -q
npm test
uv run ruff check .
uv run ruff format --check .
```

不使用 uv 时也可在独立 Python 3.11 venv 中安装 `requirements.lock.txt` 和本项目。不要把本项目的依赖装入系统 Python。

## 直接运行

已经交付完整生成数据时无需再次生成。同名数据目录和不同版本的断点目录不允许覆盖。

```powershell
# 为新实验构造独立、可复现的版本
uv run pii-eval generate --out data/generated/my-version --standard 14000 --complex 2000 --gate 1500 --seed 42
uv run pii-eval validate --data data/generated/my-version

# 默认门禁集；产出 HTML / Markdown / JSON / JSONL，及真实脱敏导出
uv run pii-eval run --data data/generated/final --model demo-student --out reports/work/student
uv run pii-eval run --data data/generated/final --model demo-teacher --out reports/work/teacher
uv run pii-eval run --data data/generated/final --model demo-routed --out reports/work/routed

# 同一批样本、同一规则配对比较。不能拿不同数据版本直接比优劣。
uv run pii-eval compare --a reports/work/student/report.json --b reports/work/teacher/report.json --out reports/work/comparison.json

# baseline 必须来自独立冻结的已批准版本。下面路径仅表示用法。
uv run pii-eval gate --report reports/work/student/report.json --baseline conf/approved-baseline.json --level L1 --out reports/work/gate.json

# 三次复现 + 五类实际SUT故障注入；合成工程验收，不需要部署服务
uv run pii-eval verify --data data/generated/final --out reports/work/verification-new

# 替代实验分别重跑金标OCR、金标布局和两者联合
uv run pii-eval attribute --data data/generated/final --out reports/work/attribution

# 最小复现和增量诊断（不具备正式PASS资格）
uv run pii-eval run --data data/generated/final --sample cx_000137 --no-cache --out reports/work/reproduce
uv run pii-eval run --data data/generated/final --layout scan --entity-type PHONE --out reports/work/incremental
```

`python -m src.runner`、`python -m src.gate` 兼容技术文档入口；统一完整命令使用 `pii-eval --help` 或 `python -m src.cli --help`。默认 `gate`，不触及留出集。最终全量验收需显式 `--split train,dev,holdout,gate --allow-holdout`，并生成留出集访问审计。

`demo-student`、`demo-teacher` 是规则工程基线，**不是训练后的 Student 或大模型 Teacher**。它们可能在合成数据上得分相同，不预设显著优劣，不套用简历中的“86%”。真实模型通过配置接入，见接入文档。

## 输出与报告

每个评测目录包含 `report.html`、`report.md`、`report.json`、`failures.jsonl`、`predictions.jsonl`、`traces.jsonl`、`run_state.json`、`checkpoints/` 和 `artifacts/`。HTML 自包含，支持轨道、版式、实体、错误筛选、分页、CSV 导出、字符区间对照、六阶段指标与原文件/脱敏文件链接。原文只存本地，不自动发布或上传。

缓存按实际输入、源文件摘要、模型配置、规则、数据、代码和配置指纹隔离，并校验导出文件；读取缓存保留原始推理耗时。`--no-cache` 同时关闭缓存及断点复用。

主指标 exact：类型和起止字符均正确才计 TP。Partial 同时计 FP/FN，单列用于边界分析。标准文本和复杂文件各自报告，不能合并为对外总 F1。高风险宏召回及每个高风险类别分别检查；缺失/NaN/执行失败/覆盖不足阻断。合成数据可以得到工程门禁 PASS，但 `release_ready` 始终为 false。

## 复核与持续改进

```powershell
uv run pii-eval quality --data data/generated/final --out reports/work/quality-plan.json
uv run pii-eval review-queue --report reports/work/student/report.json --db runtime/feedback.sqlite
uv run pii-eval feedback list --db runtime/feedback.sqlite
uv run pii-eval feedback record --case CASE_ID --reviewer annotator-a --corrected data/templates/corrected.json
uv run pii-eval feedback record --case CASE_ID --reviewer annotator-b --corrected data/templates/corrected.json
uv run pii-eval feedback promote --case CASE_ID --target reviewed-v2
uv run pii-eval feedback export --target reviewed-v2 --out reports/work/approved.jsonl
uv run pii-eval apply-feedback --data data/generated/final --approved reports/work/approved.jsonl --version reviewed-v2 --out data/generated/reviewed-v2
uv run pii-eval drift --points data/templates/monitor-points.jsonl --baseline data/templates/monitor-baseline.json --out reports/work/alerts.json
```

复核状态区分原样采纳、修改后采纳、拒绝；需要至少两名独立复核人确认相同修正才可回流。训练/留出样本不能直接进入门禁。回流创建全新数据版本，不改源集；新版本必须重新冻结基线。质检抽样计划不等于已经执行的人工质检，程序不会编造 Kappa。

## CI 与可移交内容

`.github/workflows/gate.yml` 执行静态检查、Python/DOM 测试、合成数据、门禁复现及故障验收、打包。`ci/model-gate.example.yml` 是组织已有真实模型服务的门禁模板，没有部署步骤。`Dockerfile` 仅提供未来可移交的评测镜像定义，本次没有构建/运行。PostgreSQL、ClickHouse、Langfuse、真实 OCR 和模型服务需接入已有环境，本项目不安装这些服务。

当前第三方接口依据：[DeepEval 自定义指标](https://deepeval.com/guides/guides-building-custom-metrics)、[Langfuse SDK](https://python.reference.langfuse.com/langfuse)、[Langfuse 埋点](https://langfuse.com/docs/observability/sdk/instrumentation)。接口以锁定包的实际签名和测试为准，未照抄文档中已过时的 `langfuse.decorators`、`lf.score` 或 `DatasetItem.observe`。
