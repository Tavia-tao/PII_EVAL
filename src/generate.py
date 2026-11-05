"""确定性合成数据：金标来自构造位置，不由被测检测器生成。"""

import argparse
import random
from collections import Counter
from pathlib import Path

from src.io import digest, write_json, write_jsonl
from src.schema import HIGH_RISK, Sample


def _parts(index, rng, complex_file):
    english = index % 5 == 0
    values = [
        ("姓名", f"测试用户{index:05d}", "NAME"),
        ("身份证", f"990101199005{index % 28 + 1:02d}{index % 10000:04d}", "ID_CARD"),
        ("银行卡", f"00006688{index:011d}", "BANK_CARD"),
        ("手机", f"199{index:08d}", "PHONE"),
        ("邮箱", f"test{index}@example.invalid", "EMAIL"),
        ("地址", f"测试市虚构区样例路{index % 1000}号", "ADDRESS"),
        ("出生日期", f"1990-{index % 12 + 1:02d}-{index % 28 + 1:02d}", "BIRTH_DATE"),
        ("护照", f"E{index:08d}", "PASSPORT"),
        ("病历号", f"MR{index:08d}", "MEDICAL_ID"),
        ("车牌", f"京A{index % 100000:05d}", "PLATE"),
        ("机构", f"虚构样例机构{index % 101}", "ORG"),
        ("入职日期", f"2020-{index % 12 + 1:02d}-{index % 28 + 1:02d}", "DATE"),
    ]
    if english:
        translations = {
            "姓名": "Name",
            "身份证": "ID Card",
            "银行卡": "Bank Card",
            "手机": "Phone",
            "邮箱": "Email",
            "地址": "Address",
            "出生日期": "Birth Date",
            "护照": "Passport",
            "病历号": "Medical ID",
            "车牌": "Plate",
            "机构": "Organization",
            "入职日期": "Employment Date",
        }
        replacements = {
            "NAME": f"TestUser{index:05d}",
            "ADDRESS": f"No. {index} Sample Road, Test City",
            "PLATE": f"TEST{index:05d}",
            "ORG": f"Fictional Org {index % 101}",
        }
        values = [(translations[label], replacements.get(kind, value), kind) for label, value, kind in values]
    if index % 13 == 0:
        nested_address = (
            f"Test City Phone: 199{index:08d} Unit 1" if english else f"测试市联系人手机：199{index:08d}号"
        )
        values = [
            (label, nested_address if kind == "ADDRESS" else value, kind) for label, value, kind in values
        ]
    # 轮换类别、字段顺序及文档背景，避免仅覆盖固定位置。
    selected = [values[(index + j * 3) % len(values)] for j in range(4)]
    if index % 7 == 0:
        selected.extend(v for v in values if v not in selected)
    if index % 13 == 0:
        selected.extend(v for v in values if v[2] == "ADDRESS" and v not in selected)
    rng.shuffle(selected)
    lines = [
        f"合成{'文件' if complex_file else '文本'} {index:06d} · {['合同附件', '病历摘要', '简历', '客服工单', '财报附录'][index % 5]}（虚构）"
    ]
    if english:
        lines = [f"Synthetic {'document' if complex_file else 'text'} {index:06d} (fictional)"]
    spans, negatives = [], []
    for label, value, kind in selected:
        prefix = f"{label}：" if index % 3 and not english else f"{label}: "
        start = sum(len(line) + 1 for line in lines) + len(prefix)
        lines.append(prefix + value + ("先生" if kind == "NAME" and index % 11 == 0 and not english else ""))
        spans.append(
            {
                "start": start,
                "end": start + len(value),
                "type": kind,
                "text": value,
                "risk": "high" if kind in HIGH_RISK else "normal",
            }
        )
        if index % 13 == 0 and kind == "ADDRESS":
            phone = f"199{index:08d}"
            offset = start + value.index(phone)
            spans.append(
                {"start": offset, "end": offset + len(phone), "type": "PHONE", "text": phone, "risk": "high"}
            )
    for label, value, reason in [
        ("合同签订日", "2024-03-05", "contract_date"),
        ("订单号", f"00009999{index:011d}", "order_not_bank"),
    ]:
        if english:
            label = "Contract date" if reason == "contract_date" else "Order no"
        start = sum(len(line) + 1 for line in lines) + len(label) + 1
        lines.append(label + (":" if english else "：") + value)
        negatives.append(
            {
                "start": start,
                "end": start + len(value),
                "type": "DATE_NON_PII" if reason == "contract_date" else "ORDER_NON_PII",
                "text": value,
                "reason": reason,
                "critical": reason == "order_not_bank",
            }
        )
    return lines, spans, negatives


def _materialize(root, relpath, lines, layout):
    path = root / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".docx":
        from docx import Document

        doc = Document()
        if layout in {"table", "two_column_table"}:
            cols = 2 if layout == "two_column_table" else 1
            table = doc.add_table(rows=(len(lines) + cols - 1) // cols, cols=cols)
            for i, line in enumerate(lines):
                table.cell(i // cols, i % cols).text = line
        else:
            for line in lines:
                doc.add_paragraph(line)
        # 固定核心属性，保证逻辑可复现；ZIP容器时间戳不作为数据逻辑摘要。
        from datetime import datetime

        doc.core_properties.created = datetime(2026, 1, 1)
        doc.core_properties.modified = datetime(2026, 1, 1)
        doc.save(path)
        # ZIP条目时间戳固定，跨时间生成也得到相同文件摘要。
        from io import BytesIO
        from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

        raw = path.read_bytes()
        with ZipFile(BytesIO(raw)) as src, ZipFile(path, "w", ZIP_DEFLATED) as dst:
            for name in sorted(src.namelist()):
                info = ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
                info.compress_type = ZIP_DEFLATED
                dst.writestr(info, src.read(name))
    elif path.suffix == ".pdf":
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.cidfonts import UnicodeCIDFont
        from reportlab.pdfgen.canvas import Canvas

        if "STSong-Light" not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
        pdf = Canvas(str(path), invariant=1, pagesize=(900, 1000))
        pdf.setFont("STSong-Light", 11)
        for i, line in enumerate(lines):
            if layout == "cross_page" and i == len(lines) // 2:
                pdf.showPage()
                pdf.setFont("STSong-Light", 11)
            local_i = i % (len(lines) // 2) if layout == "cross_page" else i
            x = 460 if layout == "two_column" and i >= len(lines) // 2 else 36
            if layout == "two_column":
                local_i = i % (len(lines) // 2)
            y = 960 - local_i * 28
            pdf.drawString(x, y, line)
            if layout == "mixed" and i == 0:
                pdf.rect(720, 860, 100, 50)
        pdf.save()
    else:
        import os

        from PIL import Image, ImageDraw, ImageFont

        font_path = Path(os.getenv("PII_TEST_FONT", "C:/Windows/Fonts/msyh.ttc"))
        if not font_path.exists():
            candidates = [
                Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
                Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
            ]
            font_path = next((p for p in candidates if p.exists()), None)
        font = ImageFont.truetype(str(font_path), 20) if font_path else ImageFont.load_default()
        img = Image.new("RGB", (1000, 80 + 40 * len(lines)), "white")
        draw = ImageDraw.Draw(img)
        for i, line in enumerate(lines):
            draw.text((30, 20 + i * 40), line, fill="black", font=font)
        if layout == "stamp":
            draw.ellipse((680, 100, 850, 270), outline="red", width=4)
            draw.text((700, 170), "SYNTHETIC", fill="red", font=font)
        if layout == "handwritten":
            # 合成边注笔画，作为版式干扰，不声称是真实手写识别数据。
            draw.line(
                [(730, 300), (713, 310), (710, 330), (723, 341), (740, 330), (741, 313), (730, 300)],
                fill="#305c9d",
                width=3,
            )
            draw.line(
                [(755, 299), (751, 340), (755, 321), (781, 301), (755, 321), (781, 342)],
                fill="#305c9d",
                width=3,
            )
        img.save(path)


def generate_dataset(root, standard_count=14000, complex_count=2000, gate_count=1500, seed=42):
    root = Path(root)
    if (root / "manifest.json").exists():
        raise ValueError("目标数据集已存在，请使用新的版本目录，禁止覆盖留出集")
    total = standard_count + complex_count
    if min(standard_count, complex_count) < 1 or not 0 < gate_count < total:
        raise ValueError("数据量和 gate 数量不合法")
    rng = random.Random(seed)
    split_map = {}
    # 各轨独立分层，门禁总数精确；其余按整体70%训练、10%留出分配。
    gate_complex = min(complex_count - 1, round(gate_count * complex_count / total))
    for track, n, ng in [
        ("standard_text", standard_count, gate_count - gate_complex),
        ("complex_file", complex_count, gate_complex),
    ]:
        order = list(range(n))
        rng.shuffle(order)
        nh = min(round(n * 0.1), n - ng)
        nt = min(round(n * 0.7), n - ng - nh)
        splits = ["gate"] * ng + ["holdout"] * nh + ["train"] * nt + ["dev"] * (n - ng - nh - nt)
        split_map[track] = dict(zip(order, splits, strict=True))
    layouts = [
        "table",
        "two_column_table",
        "two_column",
        "scan",
        "mixed",
        "cross_page",
        "stamp",
        "handwritten",
    ]
    rows = {"standard_text": [], "complex_file": []}
    for index in range(total):
        cx = index >= standard_count
        track, local = ("complex_file", index - standard_count) if cx else ("standard_text", index)
        layout = layouts[local % len(layouts)] if cx else "pure_text"
        lines, spans, negatives = _parts(index, rng, cx)
        text = "\n".join(lines)
        blocks = [
            {
                "block_id": f"b{i}",
                "text": line,
                "order": i,
                "page": 2 if layout == "cross_page" and i >= len(lines) // 2 else 1,
            }
            for i, line in enumerate(lines)
        ]
        relpath = None
        ocr_mode = "none"
        observed = text
        if cx:
            suffix = (
                ".docx"
                if layout in {"table", "two_column_table"}
                else ".png"
                if layout in {"scan", "stamp", "handwritten"}
                else ".pdf"
            )
            relpath = f"files/cx_{local:06d}{suffix}"
            _materialize(root, relpath, lines, layout)
            if suffix == ".png":
                ocr_mode = "replay"
                # 与字符无关地确定是否损伤数字，金标不被改写。
                if local % 5 == 0:
                    observed = observed.replace("199", "l99", 1)
        d = {
            "sample_id": f"{'cx' if cx else 'st'}_{local:06d}",
            "doc_id": f"doc_{index:06d}",
            "parent_doc_id": f"parent_{index:06d}",
            "track": track,
            "source": "synthetic",
            "split": split_map[track][local],
            "doc_type": (Path(relpath).suffix[1:] if relpath else "text"),
            "layout": layout,
            "difficulty": "hard" if cx else ["easy", "medium", "hard"][index % 3],
            "lang": "en" if index % 5 == 0 else "zh_en_mixed" if index % 3 == 0 else "zh",
            "page_count": 2 if layout == "cross_page" else 1,
            "input": {
                "text": text if not cx else "",
                "file_path": relpath,
                "ocr_text": observed if ocr_mode == "replay" else None,
                "ocr_mode": ocr_mode,
                "text_blocks": blocks if ocr_mode == "replay" else [],
            },
            "gold": {
                "text": text,
                "spans": spans,
                "non_pii_candidates": negatives,
                "non_pii_hard_negatives": negatives,
                "ocr_text": text,
                "layout_blocks": blocks,
            },
            "meta": {
                "rule_version": "rule-v2.1",
                "annotator_ids": [],
                "kappa": None,
                "quality_source": "synthetic",
                "boundary": "nested" if index % 13 == 0 else "adhesive" if index % 11 == 0 else "clear",
            },
        }
        rows[track].append(Sample.model_validate(d).model_dump())
    root.mkdir(parents=True, exist_ok=True)
    shards = []
    for track, values in rows.items():
        path = root / f"{track}.jsonl"
        write_jsonl(path, values)
        shards.append({"path": path.name, "n": len(values), "sha256": digest(path.read_bytes())})
    all_rows = rows["standard_text"] + rows["complex_file"]
    artifacts = {
        r["input"]["file_path"]: digest((root / r["input"]["file_path"]).read_bytes())
        for r in all_rows
        if r["input"]["file_path"]
    }
    content_hash = digest({"samples": all_rows, "artifacts": artifacts})
    manifest = {
        "version": f"synthetic-v1-seed{seed}-{content_hash[:10]}",
        "n": total,
        "source": "synthetic",
        "human_verified": False,
        "seed": seed,
        "shards": shards,
        "content_hash": content_hash,
        "artifacts": artifacts,
        "counts": dict(Counter(r["split"] for r in all_rows)),
        "warning": "仅用于工程功能和鲁棒性验证，不代表客户分布、人工一致性或真实模型指标",
    }
    write_json(root / "manifest.json", manifest)
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="data/generated/final")
    p.add_argument("--standard", type=int, default=14000)
    p.add_argument("--complex", type=int, default=2000)
    p.add_argument("--gate", type=int, default=1500)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    print(generate_dataset(a.out, a.standard, a.complex, a.gate, a.seed))


if __name__ == "__main__":
    main()
