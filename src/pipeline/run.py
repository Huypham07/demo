from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.pipeline import ESGWashingPipeline


def _read_input(path: Path, ocr_mode: str, min_chars_per_page: int) -> tuple[str, dict]:
    if path.suffix.lower() in {".txt", ".md"}:
        text = path.read_text(encoding="utf-8", errors="ignore")
        return text, {"mode_used": "text_input", "page_count": 0, "char_count": len(text)}
    from src.pipeline.document_loader import load_pdf_with_docling
    out = load_pdf_with_docling(path, mode=ocr_mode, min_chars_per_page=min_chars_per_page)
    return out["text"], {k: v for k, v in out.items() if k != "text"}


def main() -> None:
    parser = argparse.ArgumentParser(description="ESG-Washing demo pipeline (single document)")
    parser.add_argument("--input", required=True, help="Path to PDF (or .txt) annual report")
    parser.add_argument("--output", default="outputs/demo", help="Output directory")
    parser.add_argument("--bank", default="DEMO_BANK", help="Bank name")
    parser.add_argument("--year", type=int, default=2024, help="Reporting year")
    parser.add_argument("--ocr-mode", default="auto", choices=["auto", "ocr", "no_ocr"])
    parser.add_argument("--min-chars-per-page", type=int, default=200)
    parser.add_argument("--config", default="config/pipeline.yml")
    parser.add_argument("--evidence-variant", default="nli", choices=["nli", "window", "no_nli"])
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/3] Loading document: {input_path}")
    text, metadata = _read_input(input_path, args.ocr_mode, args.min_chars_per_page)
    extracted_path = output_dir / "extracted.txt"
    extracted_path.write_text(text, encoding="utf-8")
    print(f"      Extracted {len(text):,} chars → {extracted_path}  (mode={metadata.get('mode_used')})")

    print(f"[2/3] Initialising pipeline (config: {args.config})")
    pipeline = ESGWashingPipeline(config_path=args.config)

    print(f"[3/3] Processing {args.bank} {args.year}")
    result = pipeline.run_single_document(
        text=text,
        bank=args.bank,
        year=args.year,
        output_dir=output_dir,
        evidence_variant=args.evidence_variant,
        metadata=metadata,
    )

    score = result["ewri_score"]
    html_path = result["report_path"]

    print()
    print(f"Done. EWRI = {score.ewri:.2f} / 100.")
    print(f"  HTML report : {html_path}")
    print(f"  JSON report : {html_path.with_suffix('.json')}")
    print(f"  Enriched data: {output_dir / 'enriched.parquet'}")


if __name__ == "__main__":
    main()
