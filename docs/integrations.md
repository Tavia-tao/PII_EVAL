# 接入已有环境

本次未部署或连接下述服务。默认 `conf/eval.yaml` 使用规则基线，Langfuse、PostgreSQL、ClickHouse、LLM judge 均关闭。`.env.example` 只列字段，不自动加载。密钥从进程环境读取，不进入配置、Git 或报告。

## 模型接口

通用检测 HTTP 接口接收：

```json
{"text":"Name: TestUser","model":"student-v3-int8","prompt_version":"p2.4"}
```

返回：

```json
{"spans":[{"start":6,"end":14,"type":"NAME","conf":0.99}],"usage":{"input":20,"output":10,"cost_usd":0.0001,"measured":true}}
```

坐标是 Unicode 码点，不能返回 UTF-16 下标或字节偏移。类型集合见 `schema.ENTITY_TYPES`。未提供成本时报告为“成本未实测”，不会把 0 元视作真实免费。温度和 Prompt 版本冻结；真实模型的权重 revision、量化方式及训练数据版本应在组织的模型注册配置中同步冻结。当前没有提供实际模型权重，不能声称完成真实蒸馏效果验证。

`models.student` 使用 HTTP 检测器。`models.teacher` 支持常见 chat-completions JSON 接口，系统提示要求返回 span，不让模型读取金标。服务应支持 `response_format=json_object`；不支持时修改对应适配层/配置后用协议契约复测。

`models.pipeline` 是整条 SUT 的 HTTP 适配器，输入包含 `sample_id/input/model_version`，返回严格的 `Prediction` 契约（六阶段、文本坐标系、时延、Token/成本、路由、错误）。文件路径必须是被测系统可访问的受控引用。线上服务需要在自己的阶段入口真实埋点；本地客户端不能替远端补造历史阶段耗时。

## OCR 与文件

`ocr_mode=none` 用于文本层文件；扫描件没有文本层时不能静默跳过。`replay` 使用显式 OCR 产物，是可复现工程回放。`tesseract` 调用已存在的 `tesseract` 可执行文件（本次没有安装），语言由 `OCR_LANGUAGE` 设置。`http` 调用已有 OCR 服务，以 multipart `file` 上传，返回 `{"blocks":[{"block_id":"b0","text":"...","page":1,"order":0,"bbox":[0,0,100,30]}]}`。bbox 必须与原文件坐标系一致，尤其是图片像素和 PDF 点坐标不可混用。

复杂布局的基础适配保留文件块与阅读顺序，可通过 gold-layout 替代实验验证损失。对于生产级跨页单元格、旋转/倾斜 OCR、矢量图文字、复杂域/批注等，推荐接入现有完整解析 SUT，并由同一评测契约测量，不宣称基础解析器已达到商业排版引擎的格式保真程度。

## Langfuse

显式开启 `langfuse.enabled=true`，提供 `LANGFUSE_BASE_URL/PUBLIC_KEY/SECRET_KEY`。锁定 SDK 5.2.0 的原生实验接口，需兼容服务版本；本次没有真实实例做连通性测试。

默认将离线镜像同步到 `pii-eval__<dataset-version>`，Dataset Item ID 稳定。只上传样本ID、输入摘要、受控文件引用和金标偏移/类型，不上传完整原文或实体值。`sync_dataset=false` 时读取已有主账本并严格比对内容摘要和样本覆盖，避免覆盖主账本。Experiment 名包含模型、规则、数据、运行 ID；每个样本有六阶段 observation，S4 有模型和 Token/成本；分别上报分数及轨道维度，结束时 flush/shutdown。

配置 `LANGFUSE_ANNOTATION_QUEUE_ID` 后，需人工复核的 trace 会加入现有复核队列。人工修正记录由本地审计账本保存；组织也可将其 `status/added/removed` 作为 human feedback score 回写，使用 reviewer 的可信身份。当前本地文件工作流不替代组织身份认证。

## 数据库

`storage.enabled=true` 时从 `DATABASE_URL` 读取 `postgresql+psycopg://...`，`pii_eval_runs` 存不可覆写的运行快照，同一 run_id 只允许相同内容幂等重试。开发测试用 SQLite，同一 SQLAlchemy 适配层；这不等于已验证 PostgreSQL 服务。

`storage.clickhouse_enabled=true` 时使用 `CLICKHOUSE_HOST/PORT/SECURE/USER/PASSWORD`，`pii_eval_metrics` 保存按轨道的指标和版本摘要，使用 ReplacingMergeTree 支持重试去重。没有在本机启动 PostgreSQL 或 ClickHouse。

## Judge

默认关闭。启用后必须指定 `judge.model` 和供应商凭据，只有无明确个人/业务锚点的日期进入 GEval 建议流程。每次运行最多 `floor(N*0.04)` 次，严格小于 5%，预算耗尽/错误/不确定统一转人工复核，不修改金标。人工抽检结论使用 `judge-calibrate` 写入校准文件；一致性不足 0.8 时下次运行禁用 judge。判断准确性和具体模型成本需在真实接入后验证。
