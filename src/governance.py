from datetime import UTC, datetime

from src.io import digest


def approve_change(kind, before, after, approval):
    required = ["author", "reviewer", "reason", "version"]
    if not all(approval.get(k) for k in required):
        raise ValueError("变更审批缺少作者、复核人、原因或生效版本")
    if approval["author"] == approval["reviewer"]:
        raise ValueError("变更需要独立复核人")
    return {
        "kind": kind,
        "before_hash": digest(before),
        "after_hash": digest(after),
        "approval": approval,
        "at": datetime.now(UTC).isoformat(),
        "note": "记录外部审批证据；人员身份真实性和签字由组织权限系统负责",
    }
