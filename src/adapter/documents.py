"""真实文件解析/导出。DOCX 保留 run 样式；PDF redaction 删除底层文本。"""

from pathlib import Path

from src.schema import Block


def _docx_paragraphs(doc):
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    def walk(parent):
        element = parent._element
        if hasattr(element, "body"):
            element = element.body
        for child in element.iterchildren():
            tag = child.tag.rsplit("}", 1)[-1]
            if tag == "p":
                yield Paragraph(child, parent)
            elif tag == "tbl":
                seen = set()
                for row in Table(child, parent).rows:
                    for cell in row.cells:
                        if cell._tc not in seen:
                            seen.add(cell._tc)
                            yield from walk(cell)

    yield from walk(doc)
    seen_parts = set()
    for section in doc.sections:
        for part in (section.header, section.footer, section.first_page_header, section.first_page_footer):
            if part.part.partname not in seen_parts:
                seen_parts.add(part.part.partname)
                yield from walk(part)


def parse_document(path) -> list[Block]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return [
            Block(block_id=f"b{i}", order=i, text=line)
            for i, line in enumerate(path.read_text(encoding="utf-8").splitlines())
        ]
    if suffix == ".docx":
        from docx import Document

        return [
            Block(block_id=f"b{i}", order=i, text=p.text, type="paragraph")
            for i, p in enumerate(p for p in _docx_paragraphs(Document(path)) if p.text)
        ]
    if suffix == ".pdf":
        import pymupdf as fitz

        result = []
        with fitz.open(path) as doc:
            for page in doc:
                for block in page.get_text("dict")["blocks"]:
                    for line in block.get("lines", []):
                        text = "".join(s["text"] for s in line["spans"])
                        if text:
                            i = len(result)
                            result.append(
                                Block(
                                    block_id=f"b{i}",
                                    order=i,
                                    page=page.number + 1,
                                    text=text,
                                    bbox=tuple(line["bbox"]),
                                )
                            )
            if not result:
                result = [
                    Block(
                        block_id=f"page{p.number}",
                        text="",
                        order=p.number,
                        page=p.number + 1,
                        bbox=tuple(p.rect),
                    )
                    for p in doc
                ]
        return result
    if suffix in {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}:
        from PIL import Image

        with Image.open(path) as img:
            return [Block(block_id="image", text="", bbox=(0, 0, img.width, img.height))]
    raise ValueError(f"不支持的文件格式: {suffix}")


def join_blocks(blocks):
    return "\n".join(b.text for b in blocks)


def mask_text(text, spans, mask="█"):
    if len(mask) != 1:
        raise ValueError("保长脱敏字符必须为单个Unicode码点")
    result = list(text)
    for s in spans:
        if not 0 <= s.start < s.end <= len(text):
            raise ValueError("脱敏边界越界")
        for i in range(s.start, s.end):
            if result[i] not in "\n\r":
                result[i] = mask
    return "".join(result)


def export_document(source, blocks, spans, destination):
    source, destination = Path(source) if source else None, Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source and source.resolve() == destination.resolve():
        raise ValueError("不得覆盖原文件")
    text = join_blocks(blocks)
    masked = mask_text(text, spans)
    suffix = destination.suffix.lower()
    if suffix == ".txt":
        destination.write_text(masked, encoding="utf-8")
    elif suffix == ".docx" and source:
        from docx import Document

        doc = Document(source)
        paragraphs = [p for p in _docx_paragraphs(doc) if p.text]
        # 使用原文段落定位，不依赖经布局重排后的块顺序。
        masks = {}
        cursor = 0
        for b in blocks:
            masks.setdefault(b.block_id, masked[cursor : cursor + len(b.text)])
            cursor += len(b.text) + 1
        for i, p in enumerate(paragraphs):
            value = masks.get(f"b{i}", p.text)
            offset = 0
            for run in p.runs:
                size = len(run.text)
                run.text = value[offset : offset + size]
                offset += size
            if offset != len(value):
                raise ValueError("DOCX 含未支持的超链接/域文本，需要专用回填适配器")
        doc.save(destination)
    elif suffix == ".pdf" and source:
        import pymupdf as fitz

        with fitz.open(source) as doc:
            cursor = 0
            for b in blocks:
                overlaps = [
                    (max(s.start - cursor, 0), min(s.end - cursor, len(b.text)))
                    for s in spans
                    if max(s.start, cursor) < min(s.end, cursor + len(b.text))
                ]
                page = doc[b.page - 1]
                for start, end in overlaps:
                    # search_for 仅在该块区域内定位；找不到必须失败，不能导出明文。
                    rects = page.search_for(b.text[start:end], clip=fitz.Rect(b.bbox) if b.bbox else None)
                    if not rects:
                        if not b.bbox:
                            raise ValueError("PDF 无可验证的脱敏坐标")
                        rects = [fitz.Rect(b.bbox)]
                    for rect in rects:
                        page.add_redact_annot(rect, fill=(0, 0, 0))
                cursor += len(b.text) + 1
            for page in doc:
                page.apply_redactions()
            doc.set_metadata({})
            doc.save(destination, garbage=4, deflate=True)
    elif suffix in {".png", ".jpg"} and source:
        from PIL import Image, ImageDraw

        with Image.open(source) as img:
            img = img.convert("RGB")
            draw = ImageDraw.Draw(img)
            cursor = 0
            for b in blocks:
                hit = any(max(s.start, cursor) < min(s.end, cursor + len(b.text)) for s in spans)
                if hit:
                    if b.bbox is None:
                        raise ValueError("扫描文件缺少 OCR bbox，无法安全回填")
                    draw.rectangle(b.bbox, fill="black")
                cursor += len(b.text) + 1
            img.save(destination)
    else:
        raise ValueError("导出格式与源文件不匹配")
    # 用实际解析器复开，而不是仅检查文件存在。
    if suffix in {".pdf", ".docx", ".txt"}:
        parse_document(destination)
    else:
        from PIL import Image

        with Image.open(destination) as img:
            img.verify()
    return str(destination.resolve())
