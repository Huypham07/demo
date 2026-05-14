import os
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

ESG_SENTENCES_PATH = Path("data/corpus/esg_sentences.parquet")
OUTPUT_PATH = Path("data/labels/action/llm_prelabels.parquet")

LABELS = ["Implemented", "Planning", "Indeterminate"]

SYSTEM_PROMPT = """Bạn là chuyên gia phân tích Tính Hành Động (Actionability) của ESG cho ngân hàng Việt Nam, dựa trên Thang độ nhận thức Bloom và định lượng GRI.

Phân loại câu văn sau vào MỘT trong 3 nhãn HÀNH ĐỘNG:

1. Implemented (Đã triển khai / Có kết quả cụ thể):
   - Chứa động từ quá khứ cấp độ cao (Bloom 3-6): "đã triển khai", "đã thực hiện", "đã hoàn thành", "đã đạt được", "đã giảm".
   - Hoặc CHỨA DẪN CHỨNG ĐỊNH LƯỢNG rõ ràng: %, tỷ, triệu VNĐ, tấn CO2, kWh, MWh, giờ (hours).
   - Các mốc thời gian đã xảy ra đối chiếu với thành tựu cụ thể ("trong năm 2023 thì đạt được...").

2. Planning (Kế hoạch / Mục tiêu tương lai):
   - Chứa động từ định hướng tương lai: "sẽ", "dự kiến", "kế hoạch", "định hướng", "mục tiêu".
   - Chứa mốc thời gian tương lai: "đến năm 2025", "năm 2030", "chặng đường 2025-2030".
   - Nhắc đến "lộ trình" (roadmap) hoặc mục tiêu "net zero".

3. Indeterminate (Cam kết chung chung / Thiếu minh chứng / Khoe khoang):
   - Lời hứa chung chung không có hành động cụ thể: "cam kết", "hướng tới", "chú trọng", "đang nghiên cứu", "xem xét", "từng bước", "chung tay", "đồng hành cùng".
   - Ngôn ngữ khuếch đại (Boosting / Exaggeration) nhưng THIẾU con số chứng minh: "hàng đầu", "tiên phong", "xuất sắc", "luôn luôn".
   - Ngôn ngữ phòng vệ (Hedging / Vagueness): "ngày càng", "góp phần", "phần nào", "dần dần".
   - LƯU Ý PHẠT (Penalty): Nếu một câu vừa có vẻ giống Indeterminate nhưng lại CHỨA CON SỐ ĐỊNH LƯỢNG cụ thể (vd: "đã tập huấn 500 giờ làm việc") -> KHÔNG được chọn Indeterminate, hãy chọn Implemented.

Trả lời ĐÚNG định dạng JSON: {"action": "<Implemented|Planning|Indeterminate>", "confidence": <0.0-1.0>, "reason": "<giải thích siêu ngắn gọn>"}"""

def create_prompt(row: pd.Series) -> str:
    context = ""
    if row.get("ctx_prev"):
        context += f"[Cau truoc]: {row['ctx_prev']}\n"
    context += f"[Cau can phan loai]: {row['sentence']}\n"
    if row.get("ctx_next"):
        context += f"[Cau sau]: {row['ctx_next']}\n"
    context += f"[ESG Topic]: {row.get('predicted_label', row.get('final_topic', 'Unknown'))}"
    return context

def call_gemini(prompt: str, client, model_name: str = "gemini-3.1-flash-lite-preview") -> dict:
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=[
                {"role": "user", "parts": [{"text": SYSTEM_PROMPT + "\n\n" + prompt}]}
            ],
            config={
                "temperature": 0.1,
                "max_output_tokens": 200,
            }
        )
        text = response.text.strip()
        if "{" in text and "}" in text:
            json_str = text[text.find("{"):text.rfind("}")+1]
            result = json.loads(json_str)
            if "action" in result:
                return result
        return {"action": "Indeterminate", "confidence": 0.5, "reason": "parse_error"}
    except Exception as e:
        return {"action": "Indeterminate", "confidence": 0.5, "reason": f"error: {str(e)[:50]}"}

def process_row(args):
    idx, row, client = args
    prompt = create_prompt(row)
    result = call_gemini(prompt, client)
    return {
        "sent_id": row["sent_id"],
        "sentence": row["sentence"],
        "ctx_prev": row.get("ctx_prev", ""),
        "ctx_next": row.get("ctx_next", ""),
        "section_title": row.get("section_title", ""),
        "block_type": row.get("block_type", ""),
        "doc_id": row.get("doc_id", ""),
        "bank": row.get("bank", ""),
        "year": row.get("year", ""),
        "topic_label": row.get("topic_label", row.get("predicted_label", "")),
        "llm_label": result.get("action", "Indeterminate"),
        "llm_confidence": result.get("confidence", 0.5),
        "llm_reason": result.get("reason", ""),
    }

def run_labeling(workers: int = 20, batch_size: int = 2000):
    """Label ALL ESG sentences with actionability labels."""
    from google import genai
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Set GOOGLE_API_KEY in .env file")
    
    client = genai.Client(api_key=api_key)
    
    print("Loading ESG sentences...")
    if not ESG_SENTENCES_PATH.exists():
        print(f"ERROR: {ESG_SENTENCES_PATH} not found.")
        print("Run topic classification pipeline first.")
        return None
    
    df = pd.read_parquet(ESG_SENTENCES_PATH)
    df = df[~df["sentence"].str.contains(r"<!-- image -->|^\d+$", regex=True, na=False)]
    df = df[df["sentence"].str.len() >= 20]
    print(f"ESG sentences to label: {len(df):,}")
    
    existing_ids = set()
    results = []
    if OUTPUT_PATH.exists():
        existing = pd.read_parquet(OUTPUT_PATH)
        existing_ids = set(existing["sent_id"].tolist())
        results = existing.to_dict("records")
        print(f"Resuming: {len(existing_ids):,} already labeled")
    
    df_todo = df[~df["sent_id"].isin(existing_ids)]
    print(f"Remaining: {len(df_todo):,}")
    
    if len(df_todo) == 0:
        print("All sentences already labeled!")
        return pd.DataFrame(results)
    
    total_batches = (len(df_todo) + batch_size - 1) // batch_size
    
    for batch_num in range(total_batches):
        start = batch_num * batch_size
        end = min(start + batch_size, len(df_todo))
        batch_df = df_todo.iloc[start:end]
        
        print(f"\n--- Batch {batch_num+1}/{total_batches} ({len(batch_df)} sentences) ---")
        
        args_list = [(idx, row, client) for idx, row in batch_df.iterrows()]
        
        batch_results = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(process_row, args): args[0] for args in args_list}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Labeling"):
                try:
                    res = future.result()
                    batch_results.append(res)
                except Exception as e:
                    print(f"Error: {e}")
        
        results.extend(batch_results)
        
        df_results = pd.DataFrame(results)
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        df_results.to_parquet(OUTPUT_PATH, index=False)
        
        dist = df_results["llm_label"].value_counts()
        print(f"Checkpoint: {len(df_results):,} total")
        print(f"Distribution:\n{dist}")
    
    df_final = pd.DataFrame(results)
    print(f"\nLABELING COMPLETE: {len(df_final):,} sentences")
    print(df_final["llm_label"].value_counts())
    print(f"Saved: {OUTPUT_PATH}")
    
    return df_final

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2000)
    args = parser.parse_args()
    
    run_labeling(workers=args.workers, batch_size=args.batch_size)
