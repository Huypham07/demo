# ESG-Washing Risk Analyzer

## Tổng quan

Pipeline nhận đầu vào là báo cáo thường niên dạng PDF hoặc văn bản thuần, sau đó:

1. Trích xuất văn bản (OCR qua Docling nếu cần)
2. Phân loại chủ đề ESG (E / S\_labor / S\_community / S\_product / G)
3. Phân loại mức độ hành động (Implemented / Planning / Indeterminate)
4. Liên kết bằng chứng hỗ trợ tuyên bố ESG (TF-IDF + NLI cross-lingual)
5. Tính chỉ số **EWRI** (ESG Washing Risk Index) ∈ [0, 100]

Hai classifier (topic + action) được finetune từ PhoBERT với **Neuro-Symbolic AI Type 4**: tri thức ký hiệu (GRI, Bloom Taxonomy, Hyland Metadiscourse) được biên dịch thành semantic loss trong quá trình huấn luyện.

## Cấu trúc thư mục

```
esg-washing/
├── config/
│   ├── pipeline.yml          # Tham số pipeline
│   └── train.yml             # Tham số huấn luyện
├── data/
│   ├── extracted/            # Văn bản thô / ZIP source PDFs
│   └── labels/               # Dữ liệu gán nhãn (topic, action)
└── src/
    ├── pipeline/
    │   ├── run.py            # CLI
    │   ├── pipeline.py       # ESGWashingPipeline
    │   ├── ewri.py           # Tính EWRI
    │   └── ...
    └── training/             # Huấn luyện mô hình phân loại
```

## Hướng dẫn chạy

### Setup môi trường

Python 3.10+, cài dependencies:

```bash
pip install -r requirements.txt
```

---

### 1. Demo — phân tích 1 báo cáo thường niên

```bash
python -m src.pipeline.run \
  --input path/to/annual_report.pdf \
  --output outputs/demo \
  --bank "TEN_NGAN_HANG" \
  --year 2024
```

| Tham số | Mô tả | Mặc định |
|---|---|---|
| `--input` | PDF hoặc `.txt` | (bắt buộc) |
| `--output` | Thư mục kết quả | `outputs/demo` |
| `--bank` | Tên ngân hàng | `DEMO_BANK` |
| `--year` | Năm báo cáo | `2024` |
| `--ocr-mode` | `auto` / `ocr` / `no_ocr` | `auto` |
| `--evidence-variant` | `nli` / `window` / `no_nli` | `nli` |

Kết quả:

```
outputs/demo/
├── extracted.txt        # Văn bản sau OCR/cleaning
├── report.html          # Báo cáo HTML (mở trình duyệt)
└── enriched.parquet     # DataFrame đầy đủ (sentence-level)
```

Report HTML gồm: điểm EWRI, phân rã thành phần, phân phối chủ đề, thống kê bằng chứng, top-10 câu rủi ro cao nhất, ví dụ tích cực.

---

### 2. Huấn luyện model

> **Pre-trained models** đã publish trên HuggingFace, tự động tải về khi chạy pipeline:
> - [`huypham71/esg-topic-classifier`](https://huggingface.co/huypham71/esg-topic-classifier)
> - [`huypham71/esg-action-classifier`](https://huggingface.co/huypham71/esg-action-classifier)

Để finetune lại từ đầu, chạy `notebooks/train-model.ipynb` trên Colab (GPU T4+), hoặc:

```bash
# Neuro-Symbolic (với semantic loss)
python -m src.training.train_model --task topic
python -m src.training.train_model --task action

# Baseline (không có semantic loss, dùng để so sánh RQ1)
python -m src.training.train_model --task topic  --no-neuro-symbolic
python -m src.training.train_model --task action --no-neuro-symbolic
```

Model được lưu vào `outputs/models/{topic,action}_classifier/final/`.

#### Tinh chỉnh siêu tham số (Optuna)

```bash
python -m src.training.tune_hyperparams --task topic  --n-trials 50
python -m src.training.tune_hyperparams --task action --n-trials 50
```

---

### 3. Liên kết bằng chứng + Grid Search tham số EWRI

Hai bước này cần chạy **trước** khi vào RQ3. Khuyến nghị chạy trên **GPU** (Colab T4+).

#### 3a. Liên kết bằng chứng (Evidence Linking)

Chạy 3 biến thể liên kết bằng chứng trên toàn bộ corpus (32K câu ESG):

```bash
python -m src.pipeline.evidence_experiments \
  --input  data/corpus/actionability_sentences.parquet \
  --output-dir outputs/experiments/evidence
```

| Biến thể | Mô tả |
|---|---|
| `nli` | TF-IDF + semantic similarity + NLI (mDeBERTa) — **dùng cho RQ3** |
| `window` | Cửa sổ lân cận ± 5 câu |
| `no_nli` | TF-IDF + semantic, không NLI |

Kết quả: `outputs/experiments/evidence/evidence_{nli,window,no_nli}.parquet`

Để chỉ chạy biến thể `nli` (nhanh hơn):
```bash
python -m src.pipeline.evidence_experiments --variants nli
```

#### 3b. Grid Search tham số EWRI

Tìm bộ tham số $(P_{\text{action}}, \lambda, C)$ tối đa hoá CV = Std/Mean của EWRI trên 50 quan sát ngân hàng-năm. Ràng buộc duy nhất là thứ tự từ luận văn:

```
P(Impl) < P(Plan) < P(Indet)     λ(Impl) > λ(Plan) > λ(Indet)     P(Indet) × C ≤ 1.0
```

```bash
python -m src.pipeline.ewri_grid_search \
  --input       outputs/experiments/evidence/evidence_nli.parquet \
  --output      outputs/ewri_grid_search_results.csv \
  --figures-dir thesis/figures
```

Kết quả:
- `outputs/ewri_grid_search_results.csv` — toàn bộ tổ hợp, sắp xếp theo CV giảm dần
- `thesis/figures/ewri_cv_distribution.{png,eps}`
- `thesis/figures/ewri_top20_params.{png,eps}`
- `thesis/figures/ewri_mean_std_scatter.{png,eps}`
- `thesis/figures/ewri_param_sensitivity.{png,eps}`

Sau khi tìm được tham số tối ưu, cập nhật vào `config/pipeline.yml`:

```yaml
ewri:
  action_penalty:
    Implemented:   <P_Impl>
    Planning:      <P_Plan>
    Indeterminate: <P_Indet>
  evidence_sensitivity:
    Implemented:   <L_Impl>
    Planning:      <L_Plan>
    Indeterminate: <L_Indet>
  contradiction_amplifier: <C>
```

---

### 4. Thực nghiệm Chương 4 (RQ1 / RQ2 / RQ3)

Mở `notebooks/c4_experiments.py` trong Jupyter hoặc VS Code (Jupyter mode).
Notebook chia thành 3 phần:

| Phần | Nội dung | Yêu cầu |
|---|---|---|
| **RQ1** | Đánh giá topic & action classifier trên test set; sinh `rq1_perclass_*.png` | model checkpoint |
| **RQ2** | Load cache 3 evidence variants; so sánh evidence rate & similarity | bước 3a đã chạy |
| **RQ3** | Tính EWRI 50 bank-year; sinh `correlation_heatmap.png`, `bank_ranking.png`, `topic_distribution.png` | bước 3a + 3b đã chạy |

Nếu chưa có `data/corpus/actionability_sentences.parquet`, chạy phần **Phụ lục** trong notebook để build corpus và chạy classify trên toàn bộ raw OCR:

```bash
# Giải nén source data trước
unzip data/extracted/raw_ocr_annual_report.zip -d data/extracted/
```

Biểu đồ được xuất ra `thesis/figures/` ở cả định dạng `.png` (150 dpi) và `.eps`.

---

### 5. Gán nhãn LLM (Gemini) — chỉ cần khi tạo data mới

```bash
python -m src.training.labeling.topic_llm_labeler \
  --input data/corpus/sentences.parquet \
  --output data/labels/topic/llm_prelabels.parquet

python -m src.training.labeling.action_llm_labeler \
  --input data/corpus/sentences.parquet \
  --output data/labels/action/llm_prelabels.parquet
```

---

## Cấu hình

### `config/pipeline.yml` — các mục quan trọng

```yaml
model:
  topic:
    hf_model_id: "huypham71/esg-topic-classifier"
    path: "outputs/models/topic_classifier/final"   # local override
    max_length: 128

  nli_model: "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"

ewri:
  action_penalty:          # P(y)
    Implemented:   0.02
    Planning:      0.15
    Indeterminate: 0.55
  evidence_sensitivity:    # λ(y)
    Implemented:   1.00
    Planning:      0.85
    Indeterminate: 0.50
  contradiction_amplifier: 1.8   # C khi NLI = contradiction
```

Model local (nếu có) luôn được ưu tiên hơn HuggingFace.
Có thể override bằng biến môi trường `MODEL_DIR` (hữu ích trên Kaggle/Colab):

```bash
MODEL_DIR=/kaggle/input/esg-models python -m src.pipeline.run ...
```

---

## Công thức EWRI

Mỗi câu ESG được tính điểm rủi ro:

$$\text{WRS} = P(y) \times \bigl(1 - \lambda(y) \times \text{support}\bigr) \times C(\text{nli})$$

- `y` ∈ {Implemented, Planning, Indeterminate}
- `support` ∈ [0, 1] — độ mạnh bằng chứng
- `C(nli)` = 1.4 nếu NLI phát hiện contradiction, ngược lại = 1.0

$$\text{EWRI} = \overline{\text{WRS}} \times 100 \in [0, 100]$$

EWRI cao = nhiều tuyên bố mơ hồ / thiếu bằng chứng / có mâu thuẫn → rủi ro ESG-Washing cao hơn.
