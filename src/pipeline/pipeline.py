import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
import yaml
from pandas import DataFrame
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    logging as hf_logging,
)

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.demo_report import generate_demo_report
from src.pipeline.evidence_extract import evidence_extract
from src.pipeline.ewri import (
    EWRIScore,
    calculate_bank_year_ewri,
    configure_ewri,
    enrich_with_risk_scores,
    print_ewri_summary,
    scores_to_dataframe,
)
from src.training.corpus.build_corpus import (
    build as build_corpus,
    build_single_document,
)
from src.training.neuro_symbolic import SymbolicReasoner

hf_logging.disable_progress_bar()

def load_pipeline_config() -> dict:
    with open("config/pipeline.yml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

class ESGWashingPipeline:
    def __init__(self):
        self.config = load_pipeline_config()
        configure_ewri(self.config.get("ewri", {}))
        self.reasoner = SymbolicReasoner(
            min_confidence=self.config.get("neuro_symbolic", {}).get("min_rule_confidence", 0.3)
        )
        self._topic_model = None
        self._action_model = None
        self._linker = None

    def _resolve_model_path(self, task: str) -> str:
        model_cfg = self.config["model"][task]
        hf_model_id = model_cfg.get("hf_model_id")
        configured_path = Path(model_cfg["path"])

        candidates: list[Path] = [configured_path, configured_path / "final"]

        seen: set[str] = set()
        for candidate in candidates:
            key = str(candidate)
            seen.add(key)

        if hf_model_id:
            print(f"Loading from HuggingFace: {hf_model_id}")
            return hf_model_id

        raise FileNotFoundError(
            f"Model for task '{task}' not found locally and no hf_model_id configured."
        )

    def _load_topic_model(self):
        if self._topic_model is not None:
            return

        model_path = self._resolve_model_path("topic")
        print(f"[Pipeline] Loading topic model: {model_path}")

        self._topic_tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        self._topic_model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self._topic_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._topic_model.to(self._topic_device)
        self._topic_model.eval()

        self._topic_id2label = self._topic_model.config.id2label

    def _load_action_model(self):
        if self._action_model is not None:
            return

        model_path = self._resolve_model_path("actionability")
        print(f"[Pipeline] Loading actionability model: {model_path}")

        self._action_tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        self._action_model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self._action_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._action_model.to(self._action_device)
        self._action_model.eval()

        self._action_id2label = self._action_model.config.id2label

    def build_corpus(self, raw_txt_path: Optional[str] = None) -> pd.DataFrame:
        print("\n" + "=" * 60)
        print("Build Corpus")
        print("=" * 60)
        build_corpus(
            input_path=raw_txt_path,
            output_blocks=self.config["paths"]["blocks"],
            output_sentences=self.config["paths"]["sentences"],
        )

        sentences_path = Path(self.config["paths"]["sentences"])
        df = pd.read_parquet(sentences_path)
        print(f"Corpus built: {len(df):,} sentences")
        return df

    def topic_classification(self, df: pd.DataFrame, batch_size: int = 64) -> pd.DataFrame:
        print("\n" + "=" * 60)
        print("Topic Classification")
        print("=" * 60)

        self._load_topic_model()

        sentences = df["sentence"].tolist()
        all_labels = []
        all_probs = []

        for i in tqdm(range(0, len(sentences), batch_size), desc="Topic classification"):
            batch = sentences[i:i + batch_size]

            inputs = self._topic_tokenizer(
                batch, return_tensors="pt", truncation=True,
                padding=True, max_length=self.config["model"]["topic"]["max_length"],
            )
            inputs = {k: v.to(self._topic_device) for k, v in inputs.items()}

            with torch.no_grad():
                logits = self._topic_model(**inputs).logits

            probs = torch.softmax(logits, dim=-1)
            pred_ids = torch.argmax(probs, dim=-1).tolist()
            confs = probs.max(dim=-1).values.tolist()
            for pred_id, conf in zip(pred_ids, confs):
                all_labels.append(self._topic_id2label[pred_id])
                all_probs.append(conf)

        df = df.copy()
        df["topic_label"] = all_labels
        df["topic_confidence"] = all_probs

        print(f"\nTopic distribution:")
        print(df["topic_label"].value_counts())

        return df

    def actionability_classification(self, df: pd.DataFrame, batch_size: int = 64) -> pd.DataFrame:
        print("\n" + "=" * 60)
        print("Actionability Classification")
        print("=" * 60)

        esg_df = df[df["topic_label"] != "Non_ESG"].copy()
        print(f"ESG sentences: {len(esg_df):,} / {len(df):,}")

        self._load_action_model()

        sentences = esg_df["sentence"].tolist()
        all_labels = []
        all_probs = []

        for i in tqdm(range(0, len(sentences), batch_size), desc="Actionability"):
            batch = sentences[i:i + batch_size]

            inputs = self._action_tokenizer(
                batch, return_tensors="pt", truncation=True,
                padding=True, max_length=self.config["model"]["actionability"]["max_length"],
            )
            inputs = {k: v.to(self._action_device) for k, v in inputs.items()}

            with torch.no_grad():
                logits = self._action_model(**inputs).logits

            probs = torch.softmax(logits, dim=-1)
            pred_ids = torch.argmax(probs, dim=-1).tolist()
            confs = probs.max(dim=-1).values.tolist()
            for pred_id, conf in zip(pred_ids, confs):
                all_labels.append(self._action_id2label[pred_id])
                all_probs.append(conf)

        esg_df["action_label"] = all_labels
        esg_df["action_confidence"] = all_probs

        print(f"\nActionability distribution:")
        print(esg_df["action_label"].value_counts())

        return esg_df

    def evidence_extr(
        self,
        df: pd.DataFrame,
        evidence_variant: str = "nli",
        corpus_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        print("\n" + "=" * 60)
        print(f"Evidence Detection + Linking [{evidence_variant}]")
        print("=" * 60)

        df = evidence_extract(df, variant=evidence_variant, config=self.config, corpus_df=corpus_df)

        has_ev = int(df["has_evidence"].sum()) if "has_evidence" in df.columns else 0
        print(f"Sentences with evidence: {has_ev:,} / {len(df):,} ({100*has_ev/max(len(df),1):.1f}%)")
        return df

    def ewri(self, df: pd.DataFrame) -> tuple[DataFrame, DataFrame, list[EWRIScore]]:
        print("\n" + "=" * 60)
        print("EWRI Calculation")
        print("=" * 60)

        df = enrich_with_risk_scores(df)

        scores = calculate_bank_year_ewri(df)
        df_scores = scores_to_dataframe(scores)
        df_scores = df_scores.sort_values("ewri", ascending=False)

        print_ewri_summary(df_scores)

        return df, df_scores, scores

    def run_single_document(
        self,
        text: str,
        bank: str,
        year: int,
        output_dir: Path,
        evidence_variant: str = "nli",
        metadata: Optional[dict] = None,
    ) -> dict:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        sentences_df = build_single_document(text, bank=bank, year=year)
        if sentences_df.empty:
            raise ValueError("No sentences extracted from document.")

        sentences_df = self.topic_classification(sentences_df)

        esg_check = sentences_df[sentences_df["topic_label"] != "Non_ESG"]
        if esg_check.empty:
            raise ValueError("No ESG sentences detected.")

        esg_df = self.actionability_classification(sentences_df)

        esg_df = self.evidence_extr(esg_df, evidence_variant=evidence_variant, corpus_df=sentences_df)
        esg_df, df_scores, ewri_scores = self.ewri(esg_df)

        if not ewri_scores:
            raise ValueError("EWRI calculation produced no results.")

        html_path = generate_demo_report(
            sentences_df=sentences_df,
            esg_df=esg_df,
            ewri_score=ewri_scores[0],
            output_dir=output_dir,
            bank=bank,
            year=year,
            metadata=metadata or {},
        )

        esg_df.to_parquet(output_dir / "enriched.parquet", index=False)

        return {
            "sentences_df": sentences_df,
            "esg_df": esg_df,
            "ewri_score": ewri_scores[0],
            "report_path": html_path,
        }

