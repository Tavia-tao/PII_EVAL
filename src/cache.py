import time
from pathlib import Path

from src.io import digest, read_json, write_json
from src.schema import Prediction


class PredictionCache:
    def __init__(self, root, ttl_days=7):
        self.root, self.ttl_seconds = Path(root), ttl_days * 86400

    def key(self, parts):
        return digest(parts)

    def get(self, key):
        path = self.root / (key + ".json")
        if not path.exists() or time.time() - path.stat().st_mtime > self.ttl_seconds:
            return None
        try:
            record = read_json(path)
            if record["key"] != key or digest(record["prediction"]) != record["sha256"]:
                return None
            pred = Prediction.model_validate(record["prediction"])
            # 源导出文件丢失/被修改时缓存失效。
            for name, expected in record.get("artifacts", {}).items():
                file = Path(name)
                if not file.exists() or digest(file.read_bytes()) != expected:
                    return None
            pred.cached = True
            return pred
        except (ValueError, KeyError, OSError):
            return None

    def put(self, key, prediction):
        if prediction.error:
            return
        payload = prediction.model_dump()
        artifacts = {
            s.artifact: digest(Path(s.artifact).read_bytes())
            for s in prediction.stages.values()
            if s.artifact and Path(s.artifact).is_file()
        }
        write_json(
            self.root / (key + ".json"),
            {"key": key, "prediction": payload, "sha256": digest(payload), "artifacts": artifacts},
        )
