import argparse
import copy
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.training.neuro_symbolic import SemanticLoss, create_semantic_loss

class TextDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int = 256):
        self.texts = df['text'].tolist()
        self.labels = df['label'].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long),
        }

def prepare_text(row: pd.Series, use_context_prev: bool = True, use_context_next: bool = True) -> str:
    parts = []
    if use_context_prev and row.get('ctx_prev'):
        parts.append(str(row['ctx_prev']))
    parts.append(str(row['sentence']))
    if use_context_next and row.get('ctx_next'):
        parts.append(str(row['ctx_next']))
    return ' '.join(parts)

def deep_update(base: dict, update: dict) -> dict:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base

def load_yaml_config(config_path: Path) -> dict:
    with config_path.open('r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f'Invalid config format in {config_path}: root must be a mapping')
    return data

def resolve_runtime_config(
    raw_config: dict,
    task: str | None = None,
    profile: str | None = None,
    model_name: str | None = None,
    output_dir: str | None = None,
    seed: int | None = None,
) -> dict:
    defaults = raw_config.get('defaults', {})
    common = raw_config.get('common', {})
    tasks_cfg = raw_config.get('tasks', {})

    chosen_task = task or defaults.get('task')
    if chosen_task not in tasks_cfg:
        raise ValueError(
            f"Unknown task '{chosen_task}'. Available tasks: {list(tasks_cfg.keys())}"
        )

    chosen_profile = profile or defaults.get('profile', 'default')
    task_cfg = copy.deepcopy(tasks_cfg[chosen_task])
    profiles = task_cfg.pop('profiles', {})

    if chosen_profile not in profiles:
        raise ValueError(
            f"Unknown profile '{chosen_profile}' for task '{chosen_task}'. "
            f'Available profiles: {list(profiles.keys())}'
        )

    resolved = {}
    deep_update(resolved, copy.deepcopy(common))
    deep_update(resolved, task_cfg)
    deep_update(resolved, copy.deepcopy(profiles[chosen_profile]))

    if model_name:
        resolved.setdefault('model', {})['name'] = model_name
    if output_dir:
        resolved.setdefault('paths', {})['output_dir'] = output_dir

    resolved['runtime'] = {
        'task': chosen_task,
        'profile': chosen_profile,
        'seed': seed if seed is not None else defaults.get('seed', 42),
    }
    return resolved

def load_dataframe(data_path: Path) -> pd.DataFrame:
    suffix = data_path.suffix.lower()
    if suffix == '.parquet':
        return pd.read_parquet(data_path)
    if suffix == '.csv':
        return pd.read_csv(data_path)
    raise ValueError(f'Unsupported data format: {data_path}. Use .parquet or .csv')

def encode_labels(df: pd.DataFrame, label_column: str, label2id: dict[str, int]) -> pd.DataFrame:
    if label_column not in df.columns:
        raise ValueError(f"Column '{label_column}' not found. Available columns: {list(df.columns)}")
    result = df.copy()
    result["label"] = result[label_column].map(label2id)
    result = result.dropna(subset=["label"])
    result["label"] = result["label"].astype(int)
    return result

def build_text_column(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    use_context_prev = config.get("data", {}).get("use_context_prev", True)
    use_context_next = config.get("data", {}).get("use_context_next", True)
    result = df.copy()
    result["text"] = result.apply(
        lambda row: prepare_text(row, use_context_prev=use_context_prev, use_context_next=use_context_next),
        axis=1,
    )
    return result

def _source_label_col(config: dict) -> str:
    """Return the source label column name for a task."""
    task = config.get("runtime", {}).get("task", "topic")
    return "topic_label" if task == "topic" else "action_label"

def load_train_val_data(config: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    train_df = build_text_column(load_dataframe(Path(config["paths"]["train_data"])), config)
    val_df = build_text_column(load_dataframe(Path(config["paths"]["val_data"])), config)

    labels = config["labels"]
    label2id = {name: idx for idx, name in enumerate(labels)}
    label_col = _source_label_col(config)

    train_df = encode_labels(train_df, label_col, label2id)
    val_df = encode_labels(val_df, label_col, label2id)
    return train_df, val_df, label2id

def load_test_data(config: dict, label2id: dict[str, int]) -> pd.DataFrame | None:
    test_path = Path(config["paths"].get("test_data", ""))
    if not test_path.exists():
        return None

    test_df = build_text_column(load_dataframe(test_path), config)
    source_col = _source_label_col(config)
    for col in config.get("test_label_candidates", [source_col]):
        if col in test_df.columns:
            return encode_labels(test_df, col, label2id)
    return None

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    macro_f1 = f1_score(labels, preds, average='macro')
    micro_f1 = f1_score(labels, preds, average='micro')
    return {
        'macro_f1': macro_f1,
        'micro_f1': micro_f1,
    }

def compute_class_weights(config: dict, labels: list, method: str = 'inverse') -> torch.Tensor:
    counts = Counter(labels)
    n_samples = len(labels)
    n_classes = len(config['labels'])

    def safe_count(index: int) -> int:
        return max(counts.get(index, 0), 1)

    if method == 'inverse':
        weights = [n_samples / (n_classes * safe_count(i)) for i in range(n_classes)]
    elif method == 'sqrt_inverse':
        weights = [np.sqrt(n_samples / (n_classes * safe_count(i))) for i in range(n_classes)]
    elif method == 'effective':
        beta = 0.9999
        weights = [(1 - beta) / (1 - beta**safe_count(i)) for i in range(n_classes)]
    else:
        weights = [1.0] * n_classes

    weights = torch.tensor(weights, dtype=torch.float32)
    weights = weights / weights.sum() * n_classes
    return weights

class NeuroSymbolicTrainer(Trainer):
    def __init__(
        self,
        class_weights: torch.Tensor = None,
        semantic_loss: SemanticLoss = None,
        tokenizer_ref=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.semantic_loss = semantic_loss
        self.tokenizer_ref = tokenizer_ref

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop('labels')
        outputs = model(**inputs)
        logits = outputs.logits

        if self.class_weights is not None:
            loss_fn = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        else:
            loss_fn = nn.CrossEntropyLoss()

        ce_loss = loss_fn(logits, labels)
        total_loss = ce_loss

        if self.semantic_loss is not None and self.tokenizer_ref is not None:
            texts = self.tokenizer_ref.batch_decode(inputs["input_ids"], skip_special_tokens=True)
            sem_loss = self.semantic_loss(logits, texts)
            total_loss = ce_loss + sem_loss

        return (total_loss, outputs) if return_outputs else total_loss

def build_training_args(config: dict, output_dir: Path, seed: int) -> TrainingArguments:
    train_cfg = config["training"]
    eval_batch_size = train_cfg.get("eval_batch_size", train_cfg["train_batch_size"] * 2)

    fp16_cfg = train_cfg.get("fp16", "auto")
    if isinstance(fp16_cfg, str) and fp16_cfg.lower() == "auto":
        fp16_cfg = torch.cuda.is_available()

    warmup_steps = int(train_cfg.get("warmup_steps", 0))
    warmup_ratio = float(train_cfg.get("warmup_ratio", 0.0))

    return TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=train_cfg["epochs"],
        per_device_train_batch_size=train_cfg["train_batch_size"],
        per_device_eval_batch_size=eval_batch_size,
        learning_rate=train_cfg["learning_rate"],
        weight_decay=train_cfg.get("weight_decay"),
        warmup_steps=warmup_steps,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type="linear",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=train_cfg.get("load_best_model_at_end", True),
        metric_for_best_model=train_cfg.get("metric_for_best_model", "macro_f1"),
        greater_is_better=True,
        logging_steps=50,
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 1),
        max_grad_norm=train_cfg.get("max_grad_norm"),
        label_smoothing_factor=train_cfg.get("label_smoothing_factor"),
        fp16=bool(fp16_cfg),
        report_to=train_cfg.get("report_to", "none"),
        seed=seed,
    )

def summarize_predictions(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> dict:
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro")),
        "report": classification_report(y_true, y_pred, target_names=labels, zero_division=0),
    }


def train_once(config: dict) -> dict:
    task = config["runtime"]["task"]
    labels = config["labels"]
    label_col = 'label'

    seed = int(config["runtime"].get("seed", 42))
    set_seed(seed)

    print("Loading labels...")
    df_train, df_val, label2id = load_train_val_data(config)
    print(f"Train: {len(df_train)}, Val: {len(df_val)}")
    print(df_train[label_col].value_counts())

    model_name = config["model"]["name"]
    max_len = config["model"]["max_length"]
    output_dir = Path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(labels),
        id2label={i: name for i, name in enumerate(labels)},
        label2id={name: i for i, name in enumerate(labels)},
        ignore_mismatched_sizes=True,
    )

    lengths = df_train['sentence'].fillna('').astype(str).apply(lambda x: len(tokenizer.tokenize(x)))
    print(lengths.describe())

    train_dataset = TextDataset(df_train, tokenizer, max_len)
    val_dataset = TextDataset(df_val, tokenizer, max_len)

    class_weights = None
    if config["training"].get("use_class_weights", False):
        class_weights = compute_class_weights(
            config,
            df_train["label"].tolist(),
            method=config["training"].get("weight_method", "inverse"),
        )

    semantic_cfg = config.get("neuro_symbolic", {})
    semantic_loss = None
    if semantic_cfg.get("enabled", True):
        semantic_loss = create_semantic_loss(task=task, labels=labels, config=config)
        print(f"Semantic Loss enabled (lambda={semantic_cfg.get('constraint_lambda', 0.3)})")
    else:
        print("Semantic Loss disabled (baseline mode)")

    callbacks = []
    early_cfg = config.get("early_stopping", {})
    if early_cfg.get("enabled", True):
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=early_cfg.get("patience", 2),
                early_stopping_threshold=early_cfg.get("threshold", 0.0),
            )
        )

    trainer = NeuroSymbolicTrainer(
        class_weights=class_weights,
        semantic_loss=semantic_loss,
        tokenizer_ref=tokenizer,
        model=model,
        args=build_training_args(config, output_dir=output_dir, seed=seed),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    print('\nTraining...')
    trainer.train()
    trainer.evaluate()

    final_path = output_dir / "final"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))

    return {
        "trainer": trainer,
        "tokenizer": tokenizer,
        "df_val": df_val,
        "label2id": label2id,
        "labels": labels,
    }

def evaluate_split(trainer: Trainer, dataset: TextDataset, labels: list[str]) -> tuple[np.ndarray, np.ndarray, dict]:
    preds = trainer.predict(dataset)
    logits = preds.predictions
    y_pred = np.argmax(logits, axis=-1)
    y_true = preds.label_ids
    return logits, y_true, summarize_predictions(y_true, y_pred, labels)

def run_pipeline(config: dict) -> dict:
    run = train_once(config)
    trainer = run["trainer"]
    tokenizer = run["tokenizer"]
    labels = run["labels"]
    task = config["runtime"]["task"]

    max_len = config["model"]["max_length"]
    val_dataset = TextDataset(run["df_val"], tokenizer, max_len)
    _, val_y_true, val_neural = evaluate_split(trainer, val_dataset, labels)

    result = {
        "mode": "neuro" if config.get("neuro_symbolic", {}).get("enabled", True) else "baseline",
        "task": task,
        "val_neural": val_neural,
    }

    test_df = load_test_data(config, run["label2id"])
    if test_df is not None and len(test_df) > 0:
        test_dataset = TextDataset(test_df, tokenizer, max_len)
        _, test_y_true, test_neural = evaluate_split(trainer, test_dataset, labels)
        result["test_neural"] = test_neural

    output_dir = Path(config["paths"]["output_dir"])
    summary_path = output_dir / "metrics_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Saved metrics summary to: {summary_path}")

    return result

def run_compare_experiment(config: dict) -> dict:
    root_output = Path(config["paths"]["output_dir"])

    baseline_cfg = copy.deepcopy(config)
    baseline_cfg.setdefault("neuro_symbolic", {})["enabled"] = False
    baseline_cfg["paths"]["output_dir"] = str(root_output / "baseline")

    neuro_cfg = copy.deepcopy(config)
    neuro_cfg.setdefault("neuro_symbolic", {})["enabled"] = True
    neuro_cfg["paths"]["output_dir"] = str(root_output / "neuro")

    print("\n=== Running baseline (no neuro-symbolic) ===")
    baseline_result = run_pipeline(baseline_cfg)

    print("\n=== Running neuro-symbolic ===")
    neuro_result = run_pipeline(neuro_cfg)

    comparison = {
        "task": config["runtime"]["task"],
        "baseline": baseline_result,
        "neuro_symbolic": neuro_result,
    }

    compare_path = root_output / "comparison_summary.json"
    compare_path.parent.mkdir(parents=True, exist_ok=True)
    with compare_path.open("w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    print(f"Saved comparison summary to: {compare_path}")

    return comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ESG classifier with optional neuro-symbolic experiment")
    parser.add_argument("--config", type=str, default="config/train.yml", help="Path to YAML config")
    parser.add_argument("--task", type=str, choices=["topic", "action"], default=None, help="Task to train")
    parser.add_argument("--profile", type=str, default=None, help="Profile name inside selected task")
    parser.add_argument("--model-name", type=str, default=None, help="Override pretrained model name")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument("--run-experiment", action="store_true", help="Train baseline + neuro and compare")
    parser.add_argument("--print-config", action="store_true", help="Print resolved config before training")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print config only")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path

    raw_config = load_yaml_config(config_path)
    run_config = resolve_runtime_config(
        raw_config=raw_config,
        task=args.task,
        profile=args.profile,
        model_name=args.model_name,
        output_dir=args.output_dir,
        seed=args.seed,
    )

    if args.print_config or args.dry_run:
        print(json.dumps(run_config, indent=2, ensure_ascii=False))

    if args.dry_run:
        print('Dry run completed. No training executed.')
        return

    if args.run_experiment:
        run_compare_experiment(run_config)
    else:
        run_pipeline(run_config)

if __name__ == '__main__':
    main()