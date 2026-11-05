from collections import Counter
from pathlib import Path

from src.io import digest, read_json, read_jsonl, write_json, write_jsonl
from src.schema import Sample


def assert_no_leakage(samples: list[Sample]):
    ids, docs, parents, hashes = {}, {}, {}, {}
    for s in samples:
        if s.sample_id in ids:
            raise ValueError(f"样本ID重复: {s.sample_id}")
        ids[s.sample_id] = s.split
        for name, value, index in [
            ("文档", s.doc_id, docs),
            ("父文档", s.parent_doc_id, parents),
            ("内容", digest(s.gold.text), hashes),
        ]:
            if value in index and (name == "内容" or index[value] != s.split):
                raise ValueError(f"{name}泄漏或重复: {s.sample_id}")
            index[value] = s.split


def load_dataset(root, *, verify=True) -> tuple[list[Sample], dict]:
    root = Path(root).resolve()
    manifest = read_json(root / "manifest.json")
    samples = []
    for shard in manifest["shards"]:
        file = (root / shard["path"]).resolve()
        if not file.is_relative_to(root):
            raise ValueError("数据清单路径越界")
        if verify and digest(file.read_bytes()) != shard["sha256"]:
            raise ValueError(f"数据摘要不一致: {file.name}")
        rows = [Sample.model_validate(d) for d in read_jsonl(file)]
        if len(rows) != shard["n"]:
            raise ValueError(f"样本数不一致: {file.name}")
        samples.extend(rows)
    if verify:
        for name, expected in manifest.get("artifacts", {}).items():
            artifact = (root / name).resolve()
            if not artifact.is_relative_to(root) or not artifact.is_file():
                raise ValueError(f"源文件路径越界或缺失: {name}")
            if digest(artifact.read_bytes()) != expected:
                raise ValueError(f"源文件摘要不一致: {name}")
        assert_no_leakage(samples)
        if len(samples) != manifest["n"]:
            raise ValueError("manifest 样本总数不一致")
    return samples, manifest


def dataset_health(samples):
    counts = Counter(e.type for s in samples for e in s.gold.spans)
    return {
        "n": len(samples),
        "by_split": dict(Counter(s.split for s in samples)),
        "by_track": dict(Counter(s.track for s in samples)),
        "by_layout": dict(Counter(s.layout for s in samples)),
        "by_type": dict(counts),
        "long_tail": [t for t, n in counts.items() if n < 50],
        "synthetic_only": all(s.source == "synthetic" for s in samples),
        "human_kappa_available": all(
            s.meta.quality_source == "human" and s.meta.kappa is not None for s in samples
        ),
    }


def register_dataset(root, version):
    """登记准备好的真实/公开数据镜像；不允许覆盖已冻结版本。"""
    root = Path(root).resolve()
    if (root / "manifest.json").exists():
        raise ValueError("manifest已存在，请使用新版本目录")
    samples, shards, artifacts = [], [], {}
    for track in ["standard_text", "complex_file"]:
        path = root / (track + ".jsonl")
        rows = [Sample.model_validate(r) for r in read_jsonl(path)]
        if any(s.track != track for s in rows):
            raise ValueError("样本轨道与JSONL文件名不一致")
        samples.extend(rows)
        shards.append({"path": path.name, "n": len(rows), "sha256": digest(path.read_bytes())})
        for s in rows:
            if s.input.file_path:
                p = (root / s.input.file_path).resolve()
                if not p.is_relative_to(root):
                    raise ValueError("文件路径越界")
                artifacts[s.input.file_path] = digest(p.read_bytes())
    assert_no_leakage(samples)
    manifest = {
        "version": version,
        "n": len(samples),
        "shards": shards,
        "artifacts": artifacts,
        "content_hash": digest({"samples": [s.model_dump() for s in samples], "artifacts": artifacts}),
        "source": "synthetic" if all(s.source == "synthetic" for s in samples) else "mixed_real",
        "human_verified": all(s.meta.quality_source == "human" for s in samples),
    }
    write_json(root / "manifest.json", manifest)
    return manifest


def apply_feedback(source_root, approved_path, target_root, version):
    """双人复核结果落入新数据版本；原始数据、训练集与留出集不可被回流改写。"""
    import shutil

    from src.schema import HIGH_RISK

    source, target = Path(source_root).resolve(), Path(target_root).resolve()
    if target.exists() or target.is_relative_to(source):
        raise ValueError("新版本目录必须不存在，且不能位于源数据集内")
    samples, manifest = load_dataset(source)
    approved = list(read_jsonl(approved_path))
    changes = {}
    by_id = {s.sample_id: s for s in samples}
    for row in approved:
        if row["target"] != version or len(set(row["reviewers"])) < 2:
            raise ValueError("目标版本错误或缺少两名独立复核人")
        sample = by_id[row["sample_id"]]
        if sample.split in {"train", "holdout"}:
            raise ValueError("训练/留出样本不得回流门禁")
        d = sample.model_dump()
        d["split"] = "gate"
        d["meta"]["annotator_ids"] = row["reviewers"]
        d["gold"]["spans"] = [
            {
                **span,
                "text": sample.gold.text[span["start"] : span["end"]],
                "risk": "high" if span["type"] in HIGH_RISK else "normal",
            }
            for span in row["corrected"]
        ]
        changes[sample.sample_id] = Sample.model_validate(d)
    updated = [changes.get(s.sample_id, s) for s in samples]
    assert_no_leakage(updated)
    target.mkdir(parents=True)
    for track in ["standard_text", "complex_file"]:
        write_jsonl(target / (track + ".jsonl"), [s.model_dump() for s in updated if s.track == track])
    for relpath in manifest.get("artifacts", {}):
        out = target / relpath
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / relpath, out)
    registered = register_dataset(target, version)
    write_json(
        target / "feedback_provenance.json",
        {
            "source_hash": manifest["content_hash"],
            "approved_hash": digest(Path(approved_path).read_bytes()),
            "modified_ids": sorted(changes),
            "version": version,
        },
    )
    return registered
