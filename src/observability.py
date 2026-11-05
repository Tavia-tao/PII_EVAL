"""离线审计镜像及 Langfuse 5.x 实验/六阶段链路适配。默认不连接外部服务。"""

import contextlib
import math
import os

from src.io import digest

SCORES = {
    "f1": "overall_f1",
    "high_risk_recall": "high_risk_recall",
    "false_delete": "false_delete",
    "false_negative": "false_negative",
    "boundary_acc": "boundary_acc",
    "risk_score": "risk_score",
    "latency_ms": "latency_ms",
    "token_cost": "token_cost",
}


class Observer:
    def __init__(self, cfg, run_id, meta):
        self.cfg, self.run_id, self.meta = cfg, run_id, meta
        self.client = None
        if cfg["enabled"]:
            from langfuse import Langfuse

            self.client = Langfuse(
                public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
                secret_key=os.environ["LANGFUSE_SECRET_KEY"],
                base_url=os.environ["LANGFUSE_BASE_URL"],
                environment="pii-evaluation",
            )

    @contextlib.contextmanager
    def sample_trace(self, sample):
        metadata = {
            **self.meta,
            "sample_id": sample.sample_id,
            "layout": sample.layout,
            "doc_type": sample.doc_type,
            "difficulty": sample.difficulty,
            "split": sample.split,
        }
        if self.client:
            from langfuse import propagate_attributes

            with propagate_attributes(
                session_id=self.run_id, tags=[sample.split, sample.track, sample.layout], metadata=metadata
            ):
                with self.client.start_as_current_observation(name="pii_pipeline", metadata=metadata) as root:
                    yield root.trace_id
        else:
            yield digest({"run": self.run_id, "sample": sample.sample_id})[:32]

    def start_stage(self, name):
        if not self.client:
            return None
        return self.client.start_observation(
            name=name,
            as_type="generation" if name == "S4_detect" else "span",
            model=self.meta["model_version"] if name == "S4_detect" else None,
        )

    @staticmethod
    def end_stage(observation, stage, usage=None):
        if observation:
            kwargs = {
                "output": stage.metrics,
                "metadata": {"elapsed_ms": stage.elapsed_ms, "artifact_ref": stage.artifact, "ok": stage.ok},
                "level": "DEFAULT" if stage.ok else "ERROR",
            }
            if usage is not None:
                kwargs["usage_details"] = {"input": usage.input, "output": usage.output}
                if usage.measured:
                    kwargs["cost_details"] = {"total": usage.cost_usd}
            observation.update(**kwargs)
            observation.end()

    def report_scores(self, trace_id, row):
        if not self.client:
            return
        for key, name in SCORES.items():
            value = row.get(key)
            if value is not None and math.isfinite(value):
                self.client.create_score(
                    trace_id=trace_id,
                    name=name,
                    value=float(value),
                    data_type="NUMERIC",
                    metadata={
                        "sample_id": row["sample_id"],
                        "stage": "S4_detect" if key in {"f1", "boundary_acc"} else "pipeline",
                        "rule_version": self.meta["rule_version"],
                        "layout": row["layout"],
                    },
                    score_id=digest([trace_id, name])[:32],
                )
        if row["needs_review"]:
            self.client.create_score(
                trace_id=trace_id,
                name="needs_human_review",
                value=1,
                data_type="BOOLEAN",
                comment=f"sample={row['sample_id']}",
            )
            queue = os.environ.get(self.cfg.get("annotation_queue_id_env", ""))
            if queue:
                self.client.api.annotation_queues.create_queue_item(
                    queue, object_id=trace_id, object_type="TRACE"
                )

    def sync_samples(self, samples, manifest):
        if not self.client:
            return []
        dataset_name = self.cfg["dataset"] + "__" + manifest["version"]
        if not self.cfg.get("sync_dataset", True):
            dataset = self.client.get_dataset(dataset_name)
            if (dataset.metadata or {}).get("content_hash") != manifest["content_hash"]:
                raise ValueError("Langfuse主账本摘要与本地镜像不一致")
            wanted = {s.sample_id for s in samples}
            items = [item for item in dataset.items if item.input.get("sample_id") in wanted]
            if len(items) != len(wanted):
                raise ValueError("Langfuse主账本缺少本次样本")
            return items
        self.client.create_dataset(
            name=dataset_name,
            metadata={
                "version": manifest["version"],
                "content_hash": manifest["content_hash"],
                "source": manifest["source"],
            },
        )
        items = []
        for sample in samples:
            # 文本与实体值留在受控本地数据账本，远端只存摘要/引用/偏移。
            item = self.client.create_dataset_item(
                dataset_name=dataset_name,
                id=digest([dataset_name, sample.sample_id])[:32],
                input={
                    "sample_id": sample.sample_id,
                    "artifact_ref": sample.input.file_path,
                    "content_hash": digest(sample.input.model_dump()),
                },
                expected_output={
                    "spans": [{"start": s.start, "end": s.end, "type": s.type} for s in sample.gold.spans]
                },
                metadata={
                    "layout": sample.layout,
                    "track": sample.track,
                    "split": sample.split,
                    "rule_version": sample.meta.rule_version,
                },
            )
            items.append(item)
        return items

    def flush(self):
        if self.client:
            self.client.flush()

    def report_run_scores(self, report, experiment_id):
        if not self.client or not experiment_id:
            return
        mapping = {
            "f1": "overall_f1",
            "high_risk_recall": "high_risk_recall",
            "false_delete": "false_delete",
            "false_negative": "false_negative",
            "boundary_acc": "boundary_acc",
            "risk_score": "risk_score",
            "latency_p95": "latency_p95",
            "token_cost": "token_cost",
        }
        for track, metrics in report["tracks"].items():
            for key, name in mapping.items():
                value = metrics.get(key)
                if (
                    value is None
                    or not math.isfinite(value)
                    or (key == "token_cost" and not metrics.get("cost_measured"))
                ):
                    continue
                self.client.create_score(
                    dataset_run_id=experiment_id,
                    name=name,
                    value=float(value),
                    data_type="NUMERIC",
                    metadata={
                        "track": track,
                        "rule_version": self.meta["rule_version"],
                        "aggregation": "micro" if key == "f1" else "run",
                    },
                    score_id=digest([experiment_id, track, name])[:32],
                )

    def close(self):
        if self.client:
            self.client.flush()
            self.client.shutdown()
