import re
import unicodedata
from pathlib import Path
import pandas as pd
from typing import Optional

try:
    from underthesea import sent_tokenize, pos_tag as _ut_pos_tag
    USE_UNDERTHESEA = True
except Exception:
    USE_UNDERTHESEA = False
    _ut_pos_tag = None

INPUT_ROOT = Path("data/extracted/raw_ocr_annual_report")
OUT_BLOCKS = Path("data/corpus/blocks.parquet")
OUT_SENTS = Path("data/corpus/sentences.parquet")

MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)

# Vietnamese abbreviations that end with a dot — prevent false sentence boundaries
_VI_ABBREVS = re.compile(
    r"\b(?:TP|HN|ĐN|PGS|GS|TS|ThS|BS|KS|CN|TH|TX|TT|NXB|Tr|tr|q|Q|No|Mr|Ms|Dr"
    r"|v\.v|v\.v\.|i\.e|e\.g|etc|vs|St|Jr|Sr)\.",
    re.IGNORECASE,
)

def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFC", s)
    s = s.replace(" ", " ")
    s = s.replace("…", ".")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def split_blocks(text: str) -> list[str]:
    return [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]

def clean_heading(text: str) -> str:
    t = text.strip()
    t = re.sub(r"^#{1,6}\s+", "", t, flags=re.MULTILINE)
    t = re.sub(r"\s*\n\s*", " ", t)
    t = t.rstrip(":").strip()
    return t

def is_md_heading(block: str) -> bool:
    first_line = block.strip().split('\n')[0]
    return bool(re.match(r"^#{1,6}\s+", first_line))

def is_meta_heading(t: str) -> bool:
    return bool(re.match(r"^[A-Z]\d{2}[/\-][A-Z0-9\-]+$", t))

def is_table_like(block: str) -> bool:
    lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
    if len(lines) < 3:
        return False
    pipe_ratio = sum(("|" in ln) for ln in lines) / len(lines)
    if pipe_ratio >= 0.5:
        return True
    colish = sum(bool(re.search(r"\S+\s{2,}\S+", ln)) for ln in lines) / len(lines)
    return colish >= 0.6

def is_bullet_like(block: str) -> bool:
    lines = [ln.lstrip() for ln in block.splitlines() if ln.strip()]
    if len(lines) < 2:
        return False
    bullet = sum(bool(re.match(r"^(\-|\•|\♦|\+)\s+", ln)) for ln in lines)
    return bullet / len(lines) >= 0.5

def _is_standalone_bullet(block: str) -> bool:
    """Single stray bullet symbol with no meaningful text."""
    return bool(re.fullmatch(r"[•\-\+♦–—]\s*", block.strip()))

def infer_block_type(block: str) -> str:
    t = block.strip()

    if is_md_heading(t) or (len(t) < 90 and t.isupper()):
        h = clean_heading(t)
        if is_meta_heading(h):
            return "meta_heading"
        return "heading_like"

    if is_table_like(block):
        return "table_like"

    if is_bullet_like(block):
        return "bullet_like"

    if re.search(r"%|tỷ|triệu|nghìn|tấn|kg|CO2|Scope|KPI", block, flags=re.IGNORECASE) and re.search(r"\d", block):
        return "kpi_like"

    return "paragraph"

def _block_ends_incomplete(block: str) -> bool:
    """True when the block ends mid-sentence (no closing punctuation)."""
    last = block.rstrip()
    if not last:
        return False
    return last[-1] not in ".!?:;»"

def _block_starts_as_continuation(block: str) -> bool:
    """True when the block looks like it continues a previous sentence."""
    first = block.lstrip()
    if not first:
        return False
    # Starts with lowercase → clear continuation
    if first[0].islower():
        return True
    # Starts with a conjunction-like word in lowercase Vietnamese
    cont_words = ("và ", "hoặc ", "nhưng ", "tuy nhiên ", "do đó ", "vì vậy ",
                  "ngoài ra ", "đồng thời ", "bên cạnh đó ")
    return any(first.lower().startswith(w) for w in cont_words)

def repair_ocr_blocks(blocks: list[str]) -> list[str]:
    """Fix two OCR artefacts common in Vietnamese PDF annual reports:
    1. Standalone bullet symbols (•, –) on their own block → drop them.
    2. Sentences broken across blocks (no end-punct → next starts lowercase) → merge.
    """
    # Pass 1: drop standalone bullets and HTML image tags
    cleaned = []
    for b in blocks:
        if _is_standalone_bullet(b):
            continue
        if re.fullmatch(r"\s*<!--\s*image\s*-->\s*", b, re.IGNORECASE):
            continue
        cleaned.append(b)

    # Pass 2: merge continuation fragments
    merged = []
    i = 0
    while i < len(cleaned):
        block = cleaned[i]
        # Greedily merge while the current block ends incomplete and next continues
        while (i + 1 < len(cleaned)
               and _block_ends_incomplete(block)
               and _block_starts_as_continuation(cleaned[i + 1])
               and infer_block_type(cleaned[i + 1]) == "paragraph"):
            i += 1
            block = block.rstrip() + " " + cleaned[i].lstrip()
        merged.append(block)
        i += 1

    return merged

def sent_split(block: str, block_type: str = "paragraph") -> list[str]:
    """Split a block into sentences, adapting strategy to block type."""
    # Tables and headings are not narrative ESG claims
    if block_type in ("table_like", "heading_like", "meta_heading"):
        return []

    # Each bullet item is its own claim unit
    if block_type == "bullet_like":
        results = []
        for ln in block.splitlines():
            ln = re.sub(r"^[\-\•\♦\+]\s+", "", ln.strip()).strip()
            if ln:
                results.append(ln)
        return results

    # Prose blocks (paragraph, kpi_like, heading_like)
    if USE_UNDERTHESEA:
        try:
            raw_sents = sent_tokenize(block)
            sents = [s.strip() for s in raw_sents if s and s.strip()]
            if sents:
                merged: list[str] = []
                for s in sents:
                    # Merge orphaned version-number continuations: "0 với..." after "4."
                    if merged and re.match(r"^\d+[\s,]", s) and re.search(r"\d\.$", merged[-1]):
                        merged[-1] = merged[-1].rstrip() + s
                    # Merge stray fragments (< 25 chars and no end-punct) into previous
                    elif merged and len(s) < 25 and not re.search(r"[.!?]$", s):
                        merged[-1] = merged[-1].rstrip() + " " + s
                    else:
                        merged.append(s)
                return merged
        except Exception:
            pass

    # Fallback regex: split on whitespace after sentence-ending punctuation
    parts = re.split(r"(?<=[.!?])\s+(?=[A-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐƠƯẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼẾỀỂỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪỬỮỰỲỴỶỸ])", block.strip())
    if len(parts) == 1:
        # No split on uppercase; try any whitespace after punctuation
        parts = re.split(r"(?<=[.!?])\s+", block.strip())
    return [p.strip() for p in parts if p.strip()]

def is_heading(block: str) -> bool:
    t = block.strip()
    if len(t) > 120:
        return False
    return t.isupper() or (t.endswith(":") and len(t) < 100)

NOISE_PATTERNS = [
    re.compile(r"<!\-\-\s*image\s*\-\->", re.IGNORECASE),
    re.compile(r"^\s*(trang|page)\s*\d+\s*$", re.IGNORECASE),
    re.compile(r"^[•\-\*\+]{1,3}\s*$"),
    re.compile(r"^[\d\.\s]+$"),
    re.compile(r"^(\.{3,}|…+|\-{3,}|_{3,}|\*{3,})$"),
    re.compile(r"^\d{1,3}$"),
]

# Compiled once at module level — was inside is_noise_sentence() and recompiled every call
_VI_VERBS_RE = re.compile(
    r"\b(là|đã|đang|sẽ|được|có|không|đạt|giảm|tăng|thực hiện|triển khai|"
    r"xây dựng|phát triển|quản lý|đảm bảo|hỗ trợ|cung cấp|áp dụng|"
    r"cam kết|ban hành|tuân thủ|tiếp nhận|tư vấn|xử lý|cho|làm|đưa|mở|"
    r"nâng|đổi|cải|tổ chức|phối hợp|kết hợp|sử dụng|đánh giá|chiếm|"
    r"đóng góp|thu hút|duy trì|hoàn thành|vượt|ghi nhận)\b",
    re.IGNORECASE,
)

# Section types that contain financial/legal/glossary boilerplate rather than ESG claims
_NOISE_SECTION_KEYWORDS = (
    "giải thưởng",
    "danh hiệu",
    "phụ lục",
    "từ viết tắt",
    "chú giải",
    "bảng chú thích",
    "rủi ro lãi suất",
    "rủi ro thanh khoản",
    "rủi ro tín dụng",
    "nghĩa vụ nợ tiềm ẩn",
)


def is_noise_sentence(s: str, section_title: str = "") -> bool:
    stripped = s.strip()

    if len(stripped) < 15:
        return True

    if section_title:
        sec_lower = section_title.lower()
        if any(kw in sec_lower for kw in _NOISE_SECTION_KEYWORDS):
            return True

    # All-caps heading / award title.
    main_text = re.sub(r"\s*\([^)]*\)", "", stripped).strip(" -–—.")
    if main_text and main_text.isupper() and len(main_text) > 10:
        return True

    # List intro phrases — incomplete without the list that follows
    if stripped.endswith(":"):
        return True

    # Glossary / abbreviation entry: "SBG – Sustainability Bond Guidelines: ..."
    # uppercase ASCII letters followed by an en/em dash or hyphen then a capital
    if re.match(r"^[A-Z]{2,8}\s*[–—\-]\s+[A-Z]", stripped):
        return True

    # TOC entries: "08 Thông điệp của Chủ tịch", "18 Thông tin chung", etc.
    if re.match(r"^\d{1,3}\s+[A-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐƠƯ]", stripped):
        return True

    # Markdown table rows and separator lines
    if stripped.startswith("|") or stripped.endswith("|"):
        return True
    if re.match(r"^[\|\-\s:=]+$", stripped):
        return True

    for pat in NOISE_PATTERNS:
        if pat.search(stripped):
            return True

    alpha_ratio = sum(c.isalpha() for c in stripped) / max(len(stripped), 1)
    if alpha_ratio < 0.3:
        return True

    # Fragment starting with dash/bullet
    if re.match(r"^[\-–—•]\s*", stripped):
        content = re.sub(r"^[\-–—•\s]+", "", stripped).strip()
        if len(content) < 40 and not _VI_VERBS_RE.search(content):
            return True

    if len(stripped) < 50 and not _VI_VERBS_RE.search(stripped):
        return True

    if USE_UNDERTHESEA and 50 <= len(stripped) < 120 and not _VI_VERBS_RE.search(stripped):
        try:
            tags = _ut_pos_tag(stripped)
            has_verb = any(tag.startswith("V") for _, tag in tags)
            if not has_verb:
                return True
        except Exception:
            pass

    return False

def _iter_txt_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() != ".txt":
            raise ValueError(f"Input file must be .txt: {input_path}")
        return [input_path]

    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    files = sorted(input_path.rglob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No .txt files found under: {input_path}")
    return files

def _extract_bank_year(
    txt_path: Path,
) -> tuple[str, int]:
    bank = txt_path.parent.name
    match = re.search(r"(19|20)\d{2}", txt_path.stem)
    if match:
        return bank, int(match.group(0))

    return bank, 0

def build_single_document(
    text: str,
    bank: str = "demo",
    year: int = 2024,
    doc_id: Optional[str] = None,
) -> pd.DataFrame:
    text = normalize_text(text)
    if doc_id is None:
        doc_id = f"{bank}_{year}_demo"

    blocks = repair_ocr_blocks(split_blocks(text))

    # collect block metadata
    _PROSE = {"paragraph", "kpi_like", "bullet_like"}
    block_metas: list[tuple[str, str]] = []
    for block in blocks:
        btype = infer_block_type(block)
        btxt = clean_heading(block) if btype in ("heading_like", "meta_heading") else block
        block_metas.append((btype, btxt))

    # create sentence rows with prev/next prose-block context
    current_section_title = "UNKNOWN"
    current_section_id = f"{doc_id}_sec0"
    sec_counter = 0
    sent_rows: list[dict] = []

    for i, (btype, block_text_clean) in enumerate(block_metas):
        if btype == "heading_like":
            sec_counter += 1
            current_section_title = block_text_clean
            current_section_id = f"{doc_id}_sec{sec_counter}"

        block_id = f"{doc_id}_b{i}"

        # Nearest prose block before/after (skip headings, tables)
        block_prev_text = ""
        for pi in range(i - 1, -1, -1):
            if block_metas[pi][0] in _PROSE:
                block_prev_text = block_metas[pi][1]
                break

        block_next_text = ""
        for ni in range(i + 1, len(block_metas)):
            if block_metas[ni][0] in _PROSE:
                block_next_text = block_metas[ni][1]
                break

        sents = sent_split(block_text_clean, block_type=btype)
        sents = [s for s in sents if len(s) >= 10]
        sents = [s for s in sents if not is_noise_sentence(s, section_title=current_section_title)]

        for j, sent in enumerate(sents):
            prev_s = sents[j - 1] if j > 0 else ""
            next_s = sents[j + 1] if j + 1 < len(sents) else ""
            sent_rows.append({
                "doc_id": doc_id,
                "bank": bank,
                "year": int(year),
                "section_id": current_section_id,
                "block_id": block_id,
                "sent_id": f"{doc_id}_s{i}_{j}",
                "sent_idx_in_block": j,
                "sentence": sent,
                "ctx_prev": prev_s,
                "ctx_next": next_s,
                "block_text": block_text_clean,
                "block_prev_text": block_prev_text,
                "block_next_text": block_next_text,
                "block_type": btype,
                "section_title": current_section_title,
            })

    return pd.DataFrame(sent_rows)

def build(
    input_path: str | Path | None = None,
    output_blocks: str | Path | None = None,
    output_sentences: str | Path | None = None
):
    input_root = Path(input_path) if input_path else INPUT_ROOT
    out_blocks = Path(output_blocks) if output_blocks else OUT_BLOCKS
    out_sents = Path(output_sentences) if output_sentences else OUT_SENTS

    block_rows = []
    sent_rows = []

    txt_files = _iter_txt_files(input_root)

    for txt_path in txt_files:
        bank_name, year_value = _extract_bank_year(txt_path)

        raw = txt_path.read_text(encoding="utf-8", errors="ignore")
        text = normalize_text(raw)
        doc_id = f"{bank_name}_{year_value}_{txt_path.stem}"

        blocks = repair_ocr_blocks(split_blocks(text))

        current_section_title = "UNKNOWN"
        current_section_id = f"{doc_id}_sec0"
        sec_counter = 0

        for i, block in enumerate(blocks):
            btype = infer_block_type(block)

            if btype == "heading_like":
                sec_counter += 1
                current_section_title = clean_heading(block)
                current_section_id = f"{doc_id}_sec{sec_counter}"

            block_id = f"{doc_id}_b{i}"

            block_text_clean = block
            if btype in ("heading_like", "meta_heading"):
                block_text_clean = clean_heading(block)

            block_rows.append({
                "doc_id": doc_id,
                "bank": bank_name,
                "year": year_value,
                "section_id": current_section_id,
                "section_title": current_section_title,
                "block_id": block_id,
                "block_type": btype,
                "block_text": block_text_clean,
                "order_in_doc": i,
                "source_path": str(txt_path),
            })

            sents = sent_split(block_text_clean, block_type=btype)
            sents = [s for s in sents if len(s) >= 10]
            sents = [s for s in sents if not is_noise_sentence(s, section_title=current_section_title)]

            for j, sent in enumerate(sents):
                prev_s = sents[j - 1] if j > 0 else ""
                next_s = sents[j + 1] if j + 1 < len(sents) else ""
                sent_rows.append({
                    "doc_id": doc_id,
                    "bank": bank_name,
                    "year": year_value,
                    "section_id": current_section_id,
                    "block_id": block_id,
                    "sent_id": f"{doc_id}_s{i}_{j}",
                    "sent_idx_in_block": j,
                    "sentence": sent,
                    "ctx_prev": prev_s,
                    "ctx_next": next_s,
                    "block_text": block_text_clean,
                    "block_type": btype,
                    "section_title": current_section_title,
                })

    df_blocks = pd.DataFrame(block_rows)
    df_sents = pd.DataFrame(sent_rows)

    out_blocks.parent.mkdir(parents=True, exist_ok=True)
    df_blocks.to_parquet(out_blocks, index=False)
    df_sents.to_parquet(out_sents, index=False)

    print(f"Blocks: {out_blocks} ({len(df_blocks):,} rows)")
    print(f"Sentences: {out_sents} ({len(df_sents):,} rows)")

    print(f"\nBlock types distribution:")
    print(df_blocks["block_type"].value_counts().to_string())

    print(f"\nDocs processed: {df_blocks['doc_id'].nunique()}")
    print(f"Banks: {sorted(df_blocks['bank'].astype(str).unique())}")

    try:
        from src.training.corpus.data_quality import run_quality_checks
        print("\n--- Running Data Quality Checks ---")
        df_checked, quality_report = run_quality_checks(
            df_sents,
            text_col="sentence",
            label_col="block_type",
            min_length=20,
            max_length=500,
        )

        cols_to_keep = [c for c in df_checked.columns if not c.startswith("_")]
        df_checked = df_checked[cols_to_keep]

        df_checked.to_parquet(out_sents, index=False)
        print(f"\nQuality summary: {quality_report['clean']}/{quality_report['total']} "
              f"sentences clean ({quality_report['clean_pct']}%)")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n[Quality check skipped: {e}]")

if __name__ == "__main__":
    build()
