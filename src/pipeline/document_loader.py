from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Optional

PAGE_NUM_RE = re.compile(r"^\s*(?:trang\s*)?\d{1,3}\s*/?\s*\d{0,3}\s*$", re.IGNORECASE)
HYPHEN_BREAK_RE = re.compile(r"-\s*\n\s*")
WHITESPACE_RE = re.compile(r"[ \t]+")
MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
IMAGE_TAG_RE = re.compile(r"<image[^>]*>", re.IGNORECASE)

def _build_converter(use_ocr: bool):
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = use_ocr
    pipeline_options.do_table_structure = True

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        }
    )

def _convert(pdf_path: Path, use_ocr: bool) -> tuple[str, int]:
    converter = _build_converter(use_ocr=use_ocr)
    result = converter.convert(str(pdf_path))
    doc = result.document
    text = doc.export_to_markdown()

    page_count = 0
    pages = getattr(doc, "pages", None)
    if pages is not None:
        try:
            page_count = len(pages)
        except TypeError:
            page_count = sum(1 for _ in pages)
    return text, page_count

def clean_extracted_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace(" ", " ").replace("​", "")
    text = HYPHEN_BREAK_RE.sub("", text)
    text = IMAGE_TAG_RE.sub("", text)

    cleaned_lines = []
    for raw_line in text.splitlines():
        line = WHITESPACE_RE.sub(" ", raw_line).strip()
        if not line:
            cleaned_lines.append("")
            continue
        if PAGE_NUM_RE.match(line):
            continue
        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)
    text = MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()

def load_pdf_with_docling(
    pdf_path: str | Path,
    mode: str = "auto",
    min_chars_per_page: int = 200,
) -> dict:
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if mode not in {"no_ocr", "ocr", "auto"}:
        raise ValueError(f"Invalid mode: {mode!r}. Use 'no_ocr' | 'ocr' | 'auto'.")

    if mode == "auto":
        text, pages = _convert(pdf_path, use_ocr=False)
        cleaned = clean_extracted_text(text)
        sufficient = pages > 0 and len(cleaned) >= pages * min_chars_per_page
        if sufficient:
            mode_used = "no_ocr"
        else:
            text, pages = _convert(pdf_path, use_ocr=True)
            cleaned = clean_extracted_text(text)
            mode_used = "ocr"
    else:
        text, pages = _convert(pdf_path, use_ocr=(mode == "ocr"))
        cleaned = clean_extracted_text(text)
        mode_used = mode

    return {
        "text": cleaned,
        "mode_used": mode_used,
        "page_count": pages,
        "char_count": len(cleaned),
    }
