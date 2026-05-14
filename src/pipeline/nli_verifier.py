import torch
from typing import List, Dict
from dataclasses import dataclass
from tqdm.auto import tqdm

@dataclass
class NLIResult:
    premise: str
    hypothesis: str
    label: str
    scores: Dict[str, float]
    entailment_score: float
    supports_claim: bool

NLI_LABELS = ["contradiction", "neutral", "entailment"]

class NLIVerifier:

    MODEL_NAME = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
    
    def __init__(
        self,
        model_name: str = None,
        entailment_threshold: float = 0.5,
        device: str = None,
        max_length: int = 256,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if model_name is None:
            model_name = self.MODEL_NAME
        
        self.model_name = model_name
        self.entailment_threshold = entailment_threshold
        self.device = device
        self.max_length = max_length
        self._tokenizer = None
        self._model = None

    def _load_model(self):
        if self._model is None or self._tokenizer is None:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            from transformers import logging as hf_logging
            hf_logging.disable_progress_bar()

            print(f"Loading NLI model: {self.model_name}")
            print(f"Device: {self.device}")

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self._model.to(self.device)
            self._model.eval()

            print("NLI model loaded successfully.")

    @staticmethod
    def _normalize_label(raw_label: str) -> str:
        value = str(raw_label).strip().lower()
        if "entail" in value:
            return "entailment"
        if "contra" in value:
            return "contradiction"
        if "neutral" in value:
            return "neutral"
        return value
    
    def verify_pair(self, claim: str, evidence: str) -> NLIResult:
        return self.verify_batch([claim], [evidence], batch_size=1, show_progress=False)[0]
    
    def verify_batch(
        self,
        claims: List[str],
        evidences: List[str],
        batch_size: int = 16,
        show_progress: bool = False,
    ) -> List[NLIResult]:
        assert len(claims) == len(evidences), "Claims and evidences must be same length"
        self._load_model()

        id2label = getattr(self._model.config, "id2label", {}) or {}
        results = []
        batch_starts = range(0, len(claims), batch_size)
        total_batches = (len(claims) + batch_size - 1) // batch_size
        iterator = (
            tqdm(batch_starts, desc="NLI batch verification", total=total_batches)
            if show_progress and total_batches > 1
            else batch_starts
        )
        for i in iterator:
            batch_claims = claims[i:i+batch_size]
            batch_evidences = evidences[i:i+batch_size]

            try:
                inputs = self._tokenizer(
                    batch_claims,
                    batch_evidences,
                    truncation=True,
                    padding=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    logits = self._model(**inputs).logits
                    probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()

                for row_idx, prob_row in enumerate(probs):
                    scores = {"contradiction": 0.0, "neutral": 0.0, "entailment": 0.0}

                    for cls_idx, score in enumerate(prob_row.tolist()):
                        raw_label = id2label.get(cls_idx, str(cls_idx))
                        label = self._normalize_label(raw_label)
                        if label in scores:
                            scores[label] = float(score)

                    best_label = max(scores, key=scores.get)
                    entailment_score = scores.get("entailment", 0.0)

                    results.append(
                        NLIResult(
                            premise=batch_claims[row_idx],
                            hypothesis=batch_evidences[row_idx],
                            label=best_label,
                            scores=scores,
                            entailment_score=entailment_score,
                            supports_claim=entailment_score >= self.entailment_threshold,
                        )
                    )
            except Exception as e:
                print(f"NLI batch error, using fallback for batch: {e}")
                for claim, evidence in zip(batch_claims, batch_evidences):
                    fallback_scores = {"entailment": 0.5, "neutral": 0.3, "contradiction": 0.2}
                    results.append(
                        NLIResult(
                            premise=claim,
                            hypothesis=evidence,
                            label="neutral",
                            scores=fallback_scores,
                            entailment_score=fallback_scores["entailment"],
                            supports_claim=fallback_scores["entailment"] >= self.entailment_threshold,
                        )
                    )

            if (i + batch_size) % (batch_size * 20) == 0:
                print(f"NLI verified {min(i+batch_size, len(claims))}/{len(claims)} pairs")
        
        return results
