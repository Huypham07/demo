import os
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

SENTENCES_PATH = Path("data/corpus/sentences.parquet")
OUTPUT_PATH = Path("data/labels/topic/llm_prelabels.parquet")

TOPICS = {
    "E": "Environmental - khi thai, nang luong, moi truong, tin dung xanh",
    "S_labor": "Social (Labor) - nhan vien, dao tao, phuc loi, an toan lao dong",
    "S_community": "Social (Community) - cong dong, tu thien, CSR, trach nhiem xa hoi",
    "S_product": "Social (Product) - khach hang, bao mat, chat luong dich vu",
    "G": "Governance - quan tri rui ro, minh bach, tuan thu, kiem soat noi bo",
    "Non_ESG": "Non-ESG - khong lien quan ESG, thong tin tai chinh thuan tuy, tieu su",
}

SYSTEM_PROMPT = """Bạn là chuyên gia phân loại ESG dành riêng cho Báo cáo Thường niên của Ngân hàng Việt Nam.

Phân loại các câu văn sau vào MỘT trong 6 nhãn dưới đây. Hãy chú ý kỹ các dấu hiệu nhận biết được đúc kết từ Bộ quy chuẩn GRI:

1. E (Environmental - Môi trường):
   - Đề cập đến: khí thải, CO2, năng lượng tái tạo, tiêu thụ năng lượng (kWh, MW), nước, rác thải, đa dạng sinh học, chuyển đổi xanh, kinh tế tuần hoàn.
   - Tài chính khí hậu: tín dụng / khoản vay xanh, trái phiếu xanh, tín chỉ carbon, tài chính bền vững.

2. S_labor (Social - Lao động):
   - Đề cập đến: tuyển dụng, giữ chân nhân tài, phúc lợi, lương, an toàn lao động, đào tạo nhân sự, lộ trình thăng tiến, bình đẳng giới.
   - KHÔNG bao gồm tiểu sử giới thiệu lãnh đạo.

3. S_community (Social - Cộng đồng):
   - Đề cập đến: cộng đồng địa phương, từ thiện, trách nhiệm xã hội (CSR), an sinh xã hội, cứu trợ, hiến máu, tài trợ giáo dục/y tế.

4. S_product (Social - Sản phẩm và Khách hàng):
   - Đề cập đến: bảo vệ người tiêu dùng, bảo mật thông tin, dữ liệu cá nhân (PDPA, Nghị định 13, PCI DSS), chất lượng dịch vụ, tài chính toàn diện.

5. G (Governance - Quản trị):
   - Đề cập đến: Ủy ban ESG, quản trị công ty, phòng chống rửa tiền (AML, KYC), chống tham nhũng, bảo vệ người tố giác, quản trị rủi ro, kiểm toán, tuân thủ pháp luật.

6. Non_ESG (Không thuộc ESG):
   - Thông tin tài chính thuần túy (lợi nhuận, tài sản), tiểu sử cá nhân của HĐQT, hoặc các thông tin chung chung không rõ ràng.

Lưu ý quan trọng giải quyết xung đột (Conflict Resolution):
- Tiểu sử của lãnh đạo/HĐQT -> BẮT BUỘC chọn Non_ESG.
- Từ "quản trị" nếu nói về cơ chế điều hành (corporate governance) -> chọn G.
- Từ "quản trị" nếu chỉ là "Cử nhân quản trị kinh doanh" trong tiểu sử -> chọn Non_ESG.

Trả lời ĐÚNG định dạng JSON: {"topic": "<NHÃN_Ở_TRÊN>", "confidence": <0.0-1.0>, "reason": "<giải thích siêu ngắn gọn>"}"""

def create_prompt(row: pd.Series) -> str:
    context = ""
    if row.get("ctx_prev"):
        context += f"[Cau truoc]: {row['ctx_prev']}\n"
    context += f"[Cau can phan loai]: {row['sentence']}\n"
    if row.get("ctx_next"):
        context += f"[Cau sau]: {row['ctx_next']}\n"
    context += f"[Section]: {row.get('section_title', '')}"
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
            if "topic" in result:
                return result
        return {"topic": "Non_ESG", "confidence": 0.5, "reason": "parse_error"}
    except Exception as e:
        return {"topic": "Non_ESG", "confidence": 0.5, "reason": f"error: {str(e)[:50]}"}

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
        "llm_label": result.get("topic", "Non_ESG"),
        "llm_confidence": result.get("confidence", 0.5),
        "llm_reason": result.get("reason", ""),
    }

def run_labeling(workers: int = 20, batch_size: int = 2000):
    from google import genai
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Set GOOGLE_API_KEY in .env file")
    
    client = genai.Client(api_key=api_key)
    
    print("Loading corpus...")
    df = pd.read_parquet(SENTENCES_PATH)
    
    df = df[~df["sentence"].str.contains(r"<!-- image -->|^\d+$", regex=True, na=False)]
    df = df[df["sentence"].str.len() >= 20]
    print(f"Sentences to label: {len(df):,}")
    
    existing_ids = set()
    results = []
    if OUTPUT_PATH.exists():
        existing = pd.read_parquet(OUTPUT_PATH)
        existing_ids = set(existing["sent_id"].tolist())
        results = existing.to_dict("records")
        print(f"Resuming from checkpoint: {len(existing_ids):,} already labeled")
    
    df_todo = df[~df["sent_id"].isin(existing_ids)]
    print(f"Remaining to label: {len(df_todo):,}")
    
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
        print(f"Checkpoint saved: {len(df_results):,} total")
        print(f"Distribution so far:\n{dist}")
    
    df_final = pd.DataFrame(results)
    print(f"\n{'='*60}")
    print(f"LABELING COMPLETE: {len(df_final):,} sentences")
    print(f"{'='*60}")
    print(df_final["llm_label"].value_counts())
    print(f"\nSaved: {OUTPUT_PATH}")
    
    return df_final

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Label ALL sentences with Gemini Flash")
    parser.add_argument("--workers", type=int, default=20, help="Parallel API workers")
    parser.add_argument("--batch-size", type=int, default=2000, help="Checkpoint save interval")
    args = parser.parse_args()
    
    run_labeling(workers=args.workers, batch_size=args.batch_size)
