"""文件模板+脚本复核流，不建设额外标注UI。所有操作都有审计事件。"""

import json
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path

from src.io import write_jsonl
from src.schema import Entity


def _canonical(spans):
    values = [Entity.model_validate(s).model_dump(exclude_none=True) for s in spans]
    return sorted(values, key=lambda s: (s["start"], s["end"], s["type"]))


class FeedbackStore:
    def __init__(self, path="runtime/feedback.sqlite"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.executescript("""
            CREATE TABLE IF NOT EXISTS cases (
              id TEXT PRIMARY KEY, sample_id TEXT, original TEXT, reason TEXT, track TEXT,
              layout TEXT, high_risk INTEGER, split TEXT, target TEXT, created_at TEXT);
            CREATE TABLE IF NOT EXISTS reviews (
              case_id TEXT, reviewer TEXT, corrected TEXT, accepted INTEGER, status TEXT, at TEXT,
              PRIMARY KEY(case_id, reviewer));
            CREATE TABLE IF NOT EXISTS audit (id INTEGER PRIMARY KEY, case_id TEXT, event TEXT, payload TEXT, at TEXT);
            """)

    def _connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        return db

    def _audit(self, db, case_id, event, payload):
        db.execute(
            "INSERT INTO audit(case_id,event,payload,at) VALUES (?,?,?,?)",
            (case_id, event, json.dumps(payload, ensure_ascii=False), datetime.now(UTC).isoformat()),
        )

    def enqueue(
        self,
        sample_id,
        original,
        reason,
        track="standard_text",
        layout="pure_text",
        high_risk=False,
        split="online",
    ):
        if split == "holdout":
            raise ValueError("留出集不得回流到反复调参的门禁集")
        cid = uuid.uuid4().hex
        with self._connect() as db:
            # 同一预测及原因幂等入队，避免每次复测产生重复任务。
            canonical = json.dumps(_canonical(original), ensure_ascii=False, sort_keys=True)
            existing = db.execute(
                "SELECT id FROM cases WHERE sample_id=? AND original=? AND reason=?",
                (sample_id, canonical, reason),
            ).fetchone()
            if existing:
                return existing["id"]
            db.execute(
                "INSERT INTO cases VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    cid,
                    sample_id,
                    canonical,
                    reason,
                    track,
                    layout,
                    int(high_risk),
                    split,
                    None,
                    datetime.now(UTC).isoformat(),
                ),
            )
            self._audit(db, cid, "enqueued", {"reason": reason})
        return cid

    def record(self, case_id, reviewer, corrected, accepted):
        if not reviewer.strip():
            raise ValueError("复核人不能为空")
        corrected = _canonical(corrected)
        with self._connect() as db:
            case = db.execute("SELECT * FROM cases WHERE id=?", (case_id,)).fetchone()
            if not case:
                raise ValueError("未知复核任务")
            if case["target"]:
                raise ValueError("已回流任务冻结，修改应创建新任务")
            original = json.loads(case["original"])
            status = (
                "rejected"
                if not accepted
                else "accepted_as_is"
                if corrected == original
                else "accepted_with_edit"
            )
            result = {
                "status": status,
                "added": [s for s in corrected if s not in original],
                "removed": [s for s in original if s not in corrected],
            }
            db.execute(
                "INSERT OR REPLACE INTO reviews VALUES (?,?,?,?,?,?)",
                (
                    case_id,
                    reviewer,
                    json.dumps(corrected, ensure_ascii=False),
                    int(accepted),
                    status,
                    datetime.now(UTC).isoformat(),
                ),
            )
            self._audit(db, case_id, "reviewed", {**result, "reviewer": reviewer})
        return result

    def promote(self, case_id, target):
        with self._connect() as db:
            case = db.execute("SELECT * FROM cases WHERE id=?", (case_id,)).fetchone()
            if not case:
                raise ValueError("未知任务")
            reviews = db.execute("SELECT * FROM reviews WHERE case_id=?", (case_id,)).fetchall()
            if (
                len(reviews) < 2
                or not all(r["accepted"] for r in reviews)
                or len({r["corrected"] for r in reviews}) != 1
            ):
                raise ValueError("需要至少两名独立复核人同意相同修正")
            if case["split"] in {"train", "holdout"}:
                raise ValueError("训练/留出样本不能直接回流门禁，以免数据泄漏")
            if case["target"] and case["target"] != target:
                raise ValueError("已回流其他版本")
            db.execute("UPDATE cases SET target=? WHERE id=?", (target, case_id))
            result = {
                "case_id": case_id,
                "sample_id": case["sample_id"],
                "target": target,
                "corrected": json.loads(reviews[0]["corrected"]),
                "reviewers": [r["reviewer"] for r in reviews],
            }
            self._audit(db, case_id, "promoted", result)
            return result

    def bad_cases(self):
        with self._connect() as db:
            rows = db.execute(
                "SELECT * FROM cases ORDER BY high_risk DESC, "
                "CASE WHEN reason='boundary' THEN 0 ELSE 1 END, created_at"
            ).fetchall()
            return [dict(r) for r in rows]

    def export_approved(self, path, target):
        rows = []
        for case in self.bad_cases():
            if case["target"] == target:
                rows.append(self.promote(case["id"], target))
        write_jsonl(path, rows)
        return {"n": len(rows), "path": str(Path(path).resolve()), "target": target}
