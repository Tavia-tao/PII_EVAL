"""PostgreSQL 运营账本、ClickHouse 分析汇总；SQLite仅用于无服务开发验证。"""

import json
import os
from datetime import UTC, datetime

from sqlalchemy import JSON, Column, MetaData, String, Table, create_engine, select


class ResultStore:
    def __init__(self, url):
        self.engine = create_engine(url, pool_pre_ping=True)
        self.metadata = MetaData()
        self.runs = Table(
            "pii_eval_runs",
            self.metadata,
            Column("run_id", String(200), primary_key=True),
            Column("created_at", String(50), nullable=False),
            Column("report", JSON, nullable=False),
        )
        self.metadata.create_all(self.engine)

    def save_run(self, run_id, report):
        with self.engine.begin() as c:
            # 同一个run只允许相同内容重试，防止结果被静默覆盖。
            existing = c.execute(
                select(self.runs.c.report).where(self.runs.c.run_id == run_id)
            ).scalar_one_or_none()
            if existing is not None:
                if existing != report:
                    raise ValueError("run_id已存在且内容不同")
                return
            c.execute(
                self.runs.insert().values(
                    run_id=run_id, report=report, created_at=datetime.now(UTC).isoformat()
                )
            )

    def get_run(self, run_id):
        with self.engine.connect() as c:
            return c.execute(select(self.runs.c.report).where(self.runs.c.run_id == run_id)).scalar_one()

    def list_runs(self):
        with self.engine.connect() as c:
            return [dict(r) for r in c.execute(select(self.runs.c.run_id, self.runs.c.created_at)).mappings()]


class ClickHouseStore:
    def __init__(self):
        import clickhouse_connect

        self.client = clickhouse_connect.get_client(
            host=os.environ["CLICKHOUSE_HOST"],
            port=int(os.getenv("CLICKHOUSE_PORT", "8443")),
            secure=os.getenv("CLICKHOUSE_SECURE", "true") == "true",
            username=os.environ.get("CLICKHOUSE_USER", "default"),
            password=os.environ["CLICKHOUSE_PASSWORD"],
        )

    def initialize(self):
        self.client.command(
            "CREATE TABLE IF NOT EXISTS pii_eval_metrics "
            "(run_id String, track String, metric String, value Nullable(Float64), meta String, "
            "recorded_at DateTime DEFAULT now()) ENGINE=ReplacingMergeTree ORDER BY (run_id,track,metric)"
        )

    def save_run(self, run_id, report):
        self.initialize()
        values = []
        for track, metrics in report["tracks"].items():
            for name, value in metrics.items():
                if value is None or isinstance(value, (int, float)):
                    values.append(
                        [run_id, track, name, value, json.dumps(report["meta"], ensure_ascii=False)]
                    )
        if values:
            self.client.insert(
                "pii_eval_metrics", values, column_names=["run_id", "track", "metric", "value", "meta"]
            )
