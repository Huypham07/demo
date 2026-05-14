"""Neuro-Symbolic Integration Module for ESG-Washing Detection."""

import re
import math
import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Callable

from src.training.labeling.grounded_rules import (
    ALL_TOPIC_RULES,
    ALL_ACTION_RULES,
    match_topic_grounded,
    match_actionability_grounded,
)

@dataclass
class Predicate:
    """
    A boolean predicate grounded on text via regex pattern matching.
    
    Semantics: predicate(text) -> truth value ∈ {True, False}
    Example: Predicate("match_CO2", [r"CO2|phát thải"]) -> True if text matches
    """
    name: str
    patterns: list[str]
    source: str = ""
    
    def evaluate(self, text: str) -> bool:
        """Evaluate this predicate on text (crisp boolean)."""
        text_lower = text.lower()
        for pattern in self.patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return True
        return False
    
    def fuzzy_evaluate(self, text: str) -> float:
        """
        Evaluate with fuzzy truth value ∈ [0, 1].
        
        More pattern matches -> higher truth value (capped at 1.0).
        This preserves differentiability while grounding in text.
        """
        text_lower = text.lower()
        match_count = 0
        for pattern in self.patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                match_count += 1
        return min(match_count * 0.5, 1.0)

@dataclass
class Constraint:
    """
    A propositional logic constraint over class probabilities.
    
    Constraints are expressed in one of these forms:
    
    1. IMPLICATION: predicate(x) -> P(label) > threshold
       "If text matches CO2 pattern, then P(E) should be high"
       
    2. NEGATION:    predicate(x) -> ¬P(label) 
       "If hedging detected, then P(Implemented) should be low"
       
    3. EXACTLY_ONE: ExactlyOne(label_1, ..., label_n)
       "Exactly one topic label should be true"
       
    4. MUTUAL_EXCLUSION: label_i -> ¬label_j
       "If topic=E, then topic≠G"
    """
    name: str
    constraint_type: str
    predicate: Optional[Predicate] = None
    target_label_idx: Optional[int] = None
    source: str = ""
    
    def __post_init__(self):
        if self.constraint_type not in ("implication", "negation", "exactly_one", "mutual_exclusion"):
            raise ValueError(f"Invalid constraint type: {self.constraint_type}")

class PropositionalKnowledgeBase:
    """
    Knowledge base of propositional logic constraints for ESG classification.
    
    Compiles GRI standards, Bloom Taxonomy, and metadiscourse theory
    into formal propositional constraints.
    
    Structure:
        KB = {α_1 ∧ α_2 ∧ ... ∧ α_n}
        where each α_i is a constraint (implication/negation/exactly_one)
    """
    
    def __init__(self, labels: list[str], task: str = "topic"):
        self.labels = labels
        self.task = task
        self.label_to_idx = {label: i for i, label in enumerate(labels)}
        self.constraints: list[Constraint] = []
        self.predicates: dict[str, list[Predicate]] = {}
        
        self._build_knowledge_base()
    
    def _build_knowledge_base(self):
        """Compile domain rules into propositional constraints."""
        
        self.constraints.append(Constraint(
            name="exactly_one",
            constraint_type="exactly_one",
            source="Classification axiom: single-label constraint",
        ))
        
        rule_source = ALL_TOPIC_RULES if self.task == "topic" else ALL_ACTION_RULES
        
        for label, rules in rule_source.items():
            if label not in self.label_to_idx:
                continue
            
            label_idx = self.label_to_idx[label]
            label_predicates = []
            
            for rule in rules:
                pred = Predicate(
                    name=rule.name,
                    patterns=rule.patterns,
                    source=rule.source,
                )
                label_predicates.append(pred)
                
                self.constraints.append(Constraint(
                    name=f"impl_{rule.name}_{label}",
                    constraint_type="implication",
                    predicate=pred,
                    target_label_idx=label_idx,
                    source=rule.source,
                ))
            
            self.predicates[label] = label_predicates
        
        if self.task == "action":
            self._build_action_negation_constraints()
    
    def _build_action_negation_constraints(self):
        """
        Build negation constraints specific to actionability.
        
        Based on Bloom Taxonomy and Hyland Metadiscourse:
        - Hedging -> ¬Implemented (hedging indicates lack of concrete action)
        - Vague commitment -> ¬Implemented
        """
        if "Implemented" not in self.label_to_idx:
            return
        
        impl_idx = self.label_to_idx["Implemented"]
        
        hedging_pred = Predicate(
            name="hedging_indicator",
            patterns=[
                r"\b(luôn|always|hướng tới|towards|nỗ lực|strive|phấn đấu)\b",
                r"\b(tiên phong|pioneering|dẫn đầu|leading|tinh thần)\b",
                r"\b(quan tâm|care about|chú trọng|focus on|cam kết|commit)\b",
            ],
            source="Hyland (2005) Metadiscourse hedging markers",
        )
        self.constraints.append(Constraint(
            name="hedge_neg_implemented",
            constraint_type="negation",
            predicate=hedging_pred,
            target_label_idx=impl_idx,
            source="Hyland (2005): hedging -> ¬Implemented",
        ))
        
        vague_pred = Predicate(
            name="vague_commitment",
            patterns=[
                r"\b(nỗ lực|cố gắng|phấn đấu|hướng tới|mong muốn)\b",
                r"\b(đóng góp vào|góp phần|thúc đẩy|đẩy mạnh)\b",
            ],
            source="Vague commitment patterns -> ¬Implemented",
        )
        self.constraints.append(Constraint(
            name="vague_neg_implemented",
            constraint_type="negation",
            predicate=vague_pred,
            target_label_idx=impl_idx,
            source="Vague commitment -> ¬Implemented",
        ))
    
    def evaluate_predicates(self, text: str) -> dict[str, float]:
        """
        Evaluate all predicates on text, return label -> fuzzy truth value.
        
        For each label, the truth value is the max fuzzy evaluation
        across all predicates associated with that label.
        """
        scores = {label: 0.0 for label in self.labels}
        
        for label, preds in self.predicates.items():
            max_truth = 0.0
            for pred in preds:
                truth = pred.fuzzy_evaluate(text)
                max_truth = max(max_truth, truth)
            scores[label] = max_truth
        
        return scores
    
    def get_active_constraints(self, text: str) -> list[tuple[Constraint, float]]:
        """
        Get constraints active for this text with their truth values.
        
        Returns list of (constraint, predicate_truth_value) pairs
        where the predicate evaluates to True/non-zero.
        """
        active = []
        
        for constraint in self.constraints:
            if constraint.constraint_type == "exactly_one":
                active.append((constraint, 1.0))
            elif constraint.predicate is not None:
                truth = constraint.predicate.fuzzy_evaluate(text)
                if truth > 0:
                    active.append((constraint, truth))
        
        return active
    
    def get_triggered_rule_names(self, text: str) -> dict[str, list[str]]:
        """Get names of triggered rules per label for explainability."""
        triggered = {label: [] for label in self.labels}
        
        for label, preds in self.predicates.items():
            for pred in preds:
                if pred.evaluate(text):
                    triggered[label].append(f"{pred.name} ({pred.source})")
        
        return triggered

class SemanticLoss(nn.Module):
    """
    Semantic Loss Function (Xu et al., ICML 2018).
    
    Implements:
        L_total = L_CE + λ × L_semantic
        
    where L_semantic = L_exactly_one + L_implications + L_negations
    
    The semantic loss encodes the propositional knowledge base
    as differentiable loss terms over the neural network's output
    probabilities.
    
    Key property: L_semantic = 0 when the output satisfies all constraints.
    This satisfies the Satisfaction axiom from Xu et al.
    """
    
    def __init__(
        self,
        knowledge_base: PropositionalKnowledgeBase,
        lambda_weight: float = 0.3,
        exactly_one_weight: float = 1.0,
        implication_weight: float = 0.5,
        negation_weight: float = 0.5,
        eps: float = 1e-8,
    ):
        """
        Args:
            knowledge_base: Compiled propositional KB
            lambda_weight: Overall weight λ for semantic loss
            exactly_one_weight: Weight for exactly-one constraint
            implication_weight: Weight for implication constraints
            negation_weight: Weight for negation constraints
            eps: Numerical stability constant
        """
        super().__init__()
        self.kb = knowledge_base
        self.lambda_weight = lambda_weight
        self.exactly_one_weight = exactly_one_weight
        self.implication_weight = implication_weight
        self.negation_weight = negation_weight
        self.eps = eps
    
    def exactly_one_loss(self, probs: torch.Tensor) -> torch.Tensor:
        """
        Exactly-one semantic loss (Xu et al., 2018, Eq. 3).
        
        L_s(exactly_one, p) = -log( Σᵢ pᵢ × Πⱼ≠ᵢ (1-pⱼ) )
        
        Efficient computation:
            Πⱼ (1-pⱼ) × Σᵢ pᵢ/(1-pᵢ)
        
        This penalizes distributions that don't concentrate
        probability mass on exactly one class.
        
        Args:
            probs: Softmax probabilities (batch_size, n_classes)
            
        Returns:
            Scalar loss (mean over batch)
        """
        probs = torch.clamp(probs, min=self.eps, max=1.0 - self.eps)
        
        one_minus_p = 1.0 - probs
        
        prod_all = torch.prod(one_minus_p, dim=1)
        
        ratio_sum = torch.sum(probs / one_minus_p, dim=1)
        
        wmc = prod_all * ratio_sum
        
        loss = -torch.log(wmc + self.eps)
        
        return loss.mean()
    
    def implication_loss(
        self,
        probs: torch.Tensor,
        texts: list[str],
    ) -> torch.Tensor:
        """
        Implication constraint loss.
        
        For constraint: predicate(x) -> P(label=k) should be high
        
        When predicate fires with truth value t:
            L_impl = -t × log(p_k)
            
        This is essentially a weighted cross-entropy where the
        "label" comes from the symbolic rule, not from annotations.
        
        Theoretical basis: This satisfies the Label-Literal
        Correspondence axiom — when a single variable is constrained,
        semantic loss reduces to cross-entropy (Xu et al., Axiom 5).
        
        Args:
            probs: Softmax probabilities (batch_size, n_classes)
            texts: Input texts for predicate evaluation
            
        Returns:
            Scalar loss (mean over batch)
        """
        batch_size = probs.size(0)
        device = probs.device
        total_loss = torch.tensor(0.0, device=device)
        active_count = 0
        
        for i, text in enumerate(texts):
            for constraint in self.kb.constraints:
                if constraint.constraint_type != "implication":
                    continue
                if constraint.predicate is None or constraint.target_label_idx is None:
                    continue
                
                truth = constraint.predicate.fuzzy_evaluate(text)
                if truth <= 0:
                    continue
                
                target_prob = probs[i, constraint.target_label_idx]
                target_prob = torch.clamp(target_prob, min=self.eps)
                total_loss = total_loss + (-truth * torch.log(target_prob))
                active_count += 1
        
        if active_count > 0:
            return total_loss / active_count
        return total_loss
    
    def negation_loss(
        self,
        probs: torch.Tensor,
        texts: list[str],
    ) -> torch.Tensor:
        """
        Negation constraint loss.
        
        For constraint: predicate(x) -> ¬P(label=k)
        When predicate fires with truth value t:
            L_neg = -t × log(1 - p_k)
        
        This penalizes high probability for the negated label.
        
        Args:
            probs: Softmax probabilities (batch_size, n_classes)
            texts: Input texts for predicate evaluation
            
        Returns:
            Scalar loss (mean over batch)
        """
        batch_size = probs.size(0)
        device = probs.device
        total_loss = torch.tensor(0.0, device=device)
        active_count = 0
        
        for i, text in enumerate(texts):
            for constraint in self.kb.constraints:
                if constraint.constraint_type != "negation":
                    continue
                if constraint.predicate is None or constraint.target_label_idx is None:
                    continue
                
                truth = constraint.predicate.fuzzy_evaluate(text)
                if truth <= 0:
                    continue
                
                target_prob = probs[i, constraint.target_label_idx]
                one_minus_p = torch.clamp(1.0 - target_prob, min=self.eps)
                total_loss = total_loss + (-truth * torch.log(one_minus_p))
                active_count += 1
        
        if active_count > 0:
            return total_loss / active_count
        return total_loss
    
    def forward(
        self,
        logits: torch.Tensor,
        texts: list[str],
    ) -> torch.Tensor:
        """
        Compute total semantic loss.
        
        L_semantic = w₁ × L_exactly_one + w₂ × L_implications + w₃ × L_negations
        
        All component losses satisfy non-negativity and satisfaction axioms.
        
        Args:
            logits: Neural network output (batch_size, n_classes)
            texts: Input texts for predicate evaluation
            
        Returns:
            λ × L_semantic (scalar)
        """
        probs = torch.softmax(logits, dim=-1)
        
        loss_eo = self.exactly_one_weight * self.exactly_one_loss(probs)
        
        loss_impl = self.implication_weight * self.implication_loss(probs, texts)
        
        loss_neg = self.negation_weight * self.negation_loss(probs, texts)
        
        total = loss_eo + loss_impl + loss_neg
        
        return self.lambda_weight * total

@dataclass
class RuleExplanation:
    """Explanation for a symbolic rule match."""
    rule_name: str
    source: str
    constraint_type: str
    truth_value: float

@dataclass
class AugmentedPrediction:
    """Prediction with neuro-symbolic augmentation and explanation."""
    label: str
    confidence: float
    neural_label: str
    neural_confidence: float
    rule_adjusted: bool
    explanations: list[str]
    rule_scores: dict[str, float]
    active_constraints: int

class ConstrainedInference:
    """
    Log-linear constrained inference for neuro-symbolic prediction.
    
    Combines neural model output with symbolic knowledge base using
    a principled probabilistic framework:
    
        P(y|x) ∝ P_neural(y|x)^(1-α) × P_rules(y|x)^α
    
    Where P_rules is derived from the knowledge base:
    - Implication constraints boost target label probability
    - Negation constraints suppress target label probability
    - Normalized to form a valid probability distribution
    
    This is consistent with the training-time semantic loss,
    ensuring no train/inference mismatch.
    
    References:
        - Ganchev, K. et al. (2010). Posterior Regularization. JMLR.
        - Hinton, G. (2002). Training Products of Experts. Neural Computation.
    """
    
    def __init__(
        self,
        knowledge_base: PropositionalKnowledgeBase,
        alpha: float = 0.3,
        eps: float = 1e-8,
    ):
        """
        Args:
            knowledge_base: Compiled propositional KB (same as training)
            alpha: Mixing weight for rules (0=pure neural, 1=pure symbolic)
            eps: Numerical stability constant
        """
        self.kb = knowledge_base
        self.alpha = alpha
        self.eps = eps
    
    def _compute_rule_log_probs(self, text: str) -> torch.Tensor:
        """
        Compute log P_rules(y|x) from the knowledge base.
        
        For each label k:
            score_k = Σ (implication truths for k) - Σ (negation truths for k)
            P_rules(k|x) ∝ exp(score_k)
        
        Returns:
            Log-probability tensor of shape (n_classes,)
        """
        n_classes = len(self.kb.labels)
        scores = torch.zeros(n_classes)
        
        for constraint in self.kb.constraints:
            if constraint.predicate is None or constraint.target_label_idx is None:
                continue
            
            truth = constraint.predicate.fuzzy_evaluate(text)
            if truth <= 0:
                continue
            
            if constraint.constraint_type == "implication":
                scores[constraint.target_label_idx] += truth
            elif constraint.constraint_type == "negation":
                scores[constraint.target_label_idx] -= truth
        
        log_probs = torch.log_softmax(scores, dim=0)
        return log_probs
    
    def predict(
        self,
        logits: torch.Tensor,
        texts: list[str],
    ) -> list[AugmentedPrediction]:
        """
        Produce constrained predictions using log-linear combination.
        
        log P(y|x) = (1-α) × log P_neural(y|x) + α × log P_rules(y|x)
        
        Args:
            logits: Neural network output (batch_size, n_classes) on any device
            texts: Input texts
            
        Returns:
            List of AugmentedPrediction with labels and explanations
        """
        logits_cpu = logits.detach().cpu()
        
        neural_log_probs = torch.log_softmax(logits_cpu, dim=-1)
        neural_probs = torch.softmax(logits_cpu, dim=-1)
        neural_preds = torch.argmax(neural_probs, dim=-1)
        neural_confs = neural_probs.max(dim=-1).values
        
        predictions = []
        
        for i, text in enumerate(texts):
            rule_log_probs = self._compute_rule_log_probs(text)
            
            combined_log_probs = (
                (1 - self.alpha) * neural_log_probs[i] +
                self.alpha * rule_log_probs
            )
            
            combined_probs = torch.softmax(combined_log_probs, dim=0)
            combined_pred = torch.argmax(combined_probs).item()
            combined_conf = combined_probs[combined_pred].item()
            
            neural_label = self.kb.labels[neural_preds[i].item()]
            combined_label = self.kb.labels[combined_pred]
            rule_adjusted = (neural_label != combined_label)
            
            active_constraints = self.kb.get_active_constraints(text)
            triggered_rules = self.kb.get_triggered_rule_names(text)
            
            explanations = []
            
            if rule_adjusted:
                explanations.append(
                    f"Posterior regularization: '{neural_label}' "
                    f"(neural={neural_confs[i]:.3f}) -> '{combined_label}' "
                    f"(combined={combined_conf:.3f})"
                )
            
            for constraint, truth in active_constraints:
                if constraint.constraint_type != "exactly_one" and truth > 0.3:
                    target = self.kb.labels[constraint.target_label_idx] if constraint.target_label_idx is not None else "?"
                    if constraint.constraint_type == "implication":
                        explanations.append(
                            f"α: {constraint.predicate.name} -> P({target})↑ "
                            f"[truth={truth:.2f}, src={constraint.source}]"
                        )
                    elif constraint.constraint_type == "negation":
                        explanations.append(
                            f"α: {constraint.predicate.name} -> ¬P({target}) "
                            f"[truth={truth:.2f}, src={constraint.source}]"
                        )
            
            if not explanations:
                explanations.append("No symbolic constraints active for this input.")
            
            rule_scores = {}
            for label in self.kb.labels:
                rule_scores[label] = round(float(torch.exp(rule_log_probs[self.kb.label_to_idx[label]])), 4)
            
            predictions.append(AugmentedPrediction(
                label=combined_label,
                confidence=combined_conf,
                neural_label=neural_label,
                neural_confidence=neural_confs[i].item(),
                rule_adjusted=rule_adjusted,
                explanations=explanations,
                rule_scores=rule_scores,
                active_constraints=len([c for c, t in active_constraints if c.constraint_type != "exactly_one"]),
            ))
        
        return predictions
    
    def predict_single(
        self, logits: torch.Tensor, text: str,
    ) -> AugmentedPrediction:
        """Convenience method for single-text prediction."""
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        return self.predict(logits, [text])[0]

class SymbolicReasoner:
    TOPIC_LABELS = ["E", "S_labor", "S_community", "S_product", "G", "Non_ESG"]
    ACTION_LABELS = ["Implemented", "Planning", "Indeterminate"]
    
    def __init__(self, min_confidence: Optional[float] = None, config: Optional[dict] = None):
        topic_ns_cfg = _resolve_neuro_symbolic_config("topic", config)
        self.min_confidence = (
            min_confidence
            if min_confidence is not None
            else float(topic_ns_cfg.get("min_confidence", 0.3))
        )
        topic_labels = _resolve_task_labels("topic", config=config, labels=None)
        action_labels = _resolve_task_labels("action", config=config, labels=None)

        self.topic_kb = PropositionalKnowledgeBase(topic_labels, "topic")
        self.action_kb = PropositionalKnowledgeBase(action_labels, "action")
    
    def get_topic_constraints(self, text: str) -> dict[str, float]:
        """Get rule score vector for topic classification."""
        return self.topic_kb.evaluate_predicates(text)
    
    def get_action_constraints(self, text: str) -> dict[str, float]:
        """Get rule score vector for actionability classification."""
        return self.action_kb.evaluate_predicates(text)
    
    def reason_topic(self, text: str, context: str = ""):
        """Apply GRI-based topic rules (backward compatible)."""
        label, confidence, matched_rules = match_topic_grounded(text, context)
        rule_scores = self.topic_kb.evaluate_predicates(text)
        triggered_rules = self.topic_kb.get_triggered_rule_names(text)
        return _SymbolicResult(
            text=text, rule_scores=rule_scores, triggered_rules=triggered_rules,
            suggested_label=label, confidence=confidence,
        )
    
    def reason_action(self, text: str, context: str = ""):
        """Apply Bloom/hedging rules (backward compatible)."""
        label, confidence, matched_rules = match_actionability_grounded(text, context)
        rule_scores = self.action_kb.evaluate_predicates(text)
        triggered_rules = self.action_kb.get_triggered_rule_names(text)
        return _SymbolicResult(
            text=text, rule_scores=rule_scores, triggered_rules=triggered_rules,
            suggested_label=label, confidence=confidence,
        )
    
    def explain_prediction(
        self, text: str, pred_label: str, task: str = "topic"
    ) -> list[str]:
        """Generate human-readable explanations for a prediction."""
        kb = self.topic_kb if task == "topic" else self.action_kb
        active = kb.get_active_constraints(text)
        triggered = kb.get_triggered_rule_names(text)
        
        explanations = []
        
        if pred_label in triggered and triggered[pred_label]:
            explanations.append(
                f"Rules SUPPORT '{pred_label}': {', '.join(triggered[pred_label][:3])}"
            )
        
        for constraint, truth in active:
            if constraint.constraint_type == "exactly_one":
                continue
            if truth < 0.3:
                continue
            target = kb.labels[constraint.target_label_idx] if constraint.target_label_idx is not None else "?"
            explanations.append(
                f"  Constraint: {constraint.name} ({constraint.constraint_type}) "
                f"-> {target} [truth={truth:.2f}]"
            )
        
        if not explanations:
            explanations.append(f"No strong symbolic rules triggered for '{pred_label}'")
        
        return explanations

@dataclass
class _SymbolicResult:
    """Internal result type for backward compatibility."""
    text: str
    rule_scores: dict[str, float]
    triggered_rules: dict[str, list[str]]
    suggested_label: str
    confidence: float

def _resolve_task_labels(task: str, config: Optional[dict], labels: Optional[list[str]]) -> list[str]:
    if labels:
        return labels

    if isinstance(config, dict):
        direct_labels = config.get("labels")
        if isinstance(direct_labels, list) and direct_labels:
            return direct_labels

        tasks_cfg = config.get("tasks")
        if isinstance(tasks_cfg, dict):
            task_cfg = tasks_cfg.get(task, {})
            task_labels = task_cfg.get("labels")
            if isinstance(task_labels, list) and task_labels:
                return task_labels

    if task == "topic":
        return SymbolicReasoner.TOPIC_LABELS
    return SymbolicReasoner.ACTION_LABELS

def _resolve_neuro_symbolic_config(task: str, config: Optional[dict]) -> dict:
    if not isinstance(config, dict):
        return {}

    if isinstance(config.get("neuro_symbolic"), dict):
        return config.get("neuro_symbolic", {})

    tasks_cfg = config.get("tasks")
    common_cfg = config.get("common")
    if isinstance(tasks_cfg, dict) and isinstance(common_cfg, dict):
        merged = dict(common_cfg.get("neuro_symbolic", {}) or {})
        task_cfg = tasks_cfg.get(task, {}) or {}
        task_ns = task_cfg.get("neuro_symbolic", {})
        if isinstance(task_ns, dict):
            merged.update(task_ns)
        return merged

    known_ns_keys = {
        "enabled",
        "constraint_lambda",
        "exactly_one_weight",
        "implication_weight",
        "negation_weight",
        "inference_alpha",
        "alpha",
        "min_confidence",
    }
    if any(key in config for key in known_ns_keys):
        return config

    return {}

def create_semantic_loss(
    task: str = "topic",
    lambda_weight: Optional[float] = None,
    exactly_one_weight: Optional[float] = None,
    implication_weight: Optional[float] = None,
    negation_weight: Optional[float] = None,
    labels: Optional[list[str]] = None,
    config: Optional[dict] = None,
) -> SemanticLoss:
    """Factory: create a SemanticLoss for the given task"""
    ns_cfg = _resolve_neuro_symbolic_config(task, config)
    resolved_labels = _resolve_task_labels(task, config=config, labels=labels)
    kb = PropositionalKnowledgeBase(resolved_labels, task)

    return SemanticLoss(
        knowledge_base=kb,
        lambda_weight=(
            float(lambda_weight)
            if lambda_weight is not None
            else float(ns_cfg.get("constraint_lambda", 0.3))
        ),
        exactly_one_weight=(
            float(exactly_one_weight)
            if exactly_one_weight is not None
            else float(ns_cfg.get("exactly_one_weight", 1.0))
        ),
        implication_weight=(
            float(implication_weight)
            if implication_weight is not None
            else float(ns_cfg.get("implication_weight", 0.5))
        ),
        negation_weight=(
            float(negation_weight)
            if negation_weight is not None
            else float(ns_cfg.get("negation_weight", 0.5))
        ),
    )

def create_constrained_inference(
    task: str = "topic",
    alpha: Optional[float] = None,
    labels: Optional[list[str]] = None,
    config: Optional[dict] = None,
) -> ConstrainedInference:
    """Factory: create a ConstrainedInference for the given task"""
    ns_cfg = _resolve_neuro_symbolic_config(task, config)
    resolved_labels = _resolve_task_labels(task, config=config, labels=labels)
    kb = PropositionalKnowledgeBase(resolved_labels, task)
    resolved_alpha = (
        float(alpha)
        if alpha is not None
        else float(ns_cfg.get("inference_alpha", ns_cfg.get("alpha", 0.3)))
    )
    return ConstrainedInference(knowledge_base=kb, alpha=resolved_alpha)

if __name__ == "__main__":
    print("=" * 60)
    print("NEURO-SYMBOLIC MODULE (Semantic Loss, Xu et al. 2018)")
    print("=" * 60)
    
    print("\n--- Propositional Knowledge Base ---")
    kb_topic = PropositionalKnowledgeBase(SymbolicReasoner.TOPIC_LABELS, "topic")
    print(f"Topic KB: {len(kb_topic.constraints)} constraints")
    for c in kb_topic.constraints[:5]:
        print(f"  {c.constraint_type}: {c.name}")
    
    kb_action = PropositionalKnowledgeBase(SymbolicReasoner.ACTION_LABELS, "action")
    print(f"Action KB: {len(kb_action.constraints)} constraints")
    
    print("\n--- Semantic Loss Test ---")
    sem_loss = create_semantic_loss("topic", lambda_weight=0.3)
    
    logits = torch.randn(3, 6)
    texts = [
        "Ngân hàng đã giảm phát thải CO2 được 15% so với năm 2022.",
        "Chúng tôi cam kết hướng tới phát triển bền vững.",
        "Đã triển khai chương trình đào tạo cho 5.000 nhân viên.",
    ]
    
    loss = sem_loss(logits, texts)
    print(f"Semantic Loss: {loss.item():.4f}")
    
    probs = torch.softmax(logits, dim=-1)
    eo_loss = sem_loss.exactly_one_loss(probs)
    print(f"  Exactly-One component: {eo_loss.item():.4f}")
    impl_loss = sem_loss.implication_loss(probs, texts)
    print(f"  Implication component: {impl_loss.item():.4f}")
    neg_loss = sem_loss.negation_loss(probs, texts)
    print(f"  Negation component: {neg_loss.item():.4f}")
    
    print("\n--- Constrained Inference Test ---")
    inferencer = create_constrained_inference("topic", alpha=0.3)
    
    preds = inferencer.predict(logits, texts)
    for pred, text in zip(preds, texts):
        print(f"\n\"{text[:60]}...\"")
        print(f"  Neural: {pred.neural_label} ({pred.neural_confidence:.3f})")
        print(f"  Final:  {pred.label} ({pred.confidence:.3f})")
        print(f"  Adjusted: {pred.rule_adjusted}")
        for exp in pred.explanations[:3]:
            print(f"    {exp}")
