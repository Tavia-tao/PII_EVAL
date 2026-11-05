import pytest


def test_reviewed_corrections_create_new_version_without_touching_source(tmp_path):
    from src.dataset import apply_feedback, load_dataset
    from src.generate import generate_dataset
    from src.io import write_jsonl

    source, target = tmp_path / "source", tmp_path / "target"
    manifest = generate_dataset(source, 60, 8, 12)
    rows, _ = load_dataset(source)
    s = next(s for s in rows if s.split == "dev" and s.track == "standard_text")
    corrected = [e.model_dump() for e in s.gold.spans[:-1]]
    approved = tmp_path / "approved.jsonl"
    write_jsonl(
        approved,
        [
            {
                "sample_id": s.sample_id,
                "corrected": corrected,
                "reviewers": ["reviewer-a", "reviewer-b"],
                "target": "reviewed-v2",
            }
        ],
    )
    apply_feedback(source, approved, target, "reviewed-v2")
    updated, new_manifest = load_dataset(target)
    assert new_manifest["version"] == "reviewed-v2"
    changed = next(r for r in updated if r.sample_id == s.sample_id)
    assert changed.split == "gate"
    assert len(changed.gold.spans) == len(corrected)
    assert load_dataset(source)[1]["content_hash"] == manifest["content_hash"]
    original = next(r for r in rows if r.split == "holdout")
    write_jsonl(
        approved,
        [{"sample_id": original.sample_id, "corrected": [], "reviewers": ["a", "b"], "target": "v3"}],
    )
    with pytest.raises(ValueError, match="留出|训练"):
        apply_feedback(source, approved, tmp_path / "forbidden", "v3")
