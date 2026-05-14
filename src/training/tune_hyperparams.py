import optuna
import json
import copy
from pathlib import Path

from src.training.train_model import (
    load_yaml_config,
    resolve_runtime_config,
    train_once,
    evaluate_split,
    TextDataset
)

def objective(trial: optuna.Trial, base_config: dict) -> float:
    """Optuna objective. Search space follows thesis Bảng tab:tuning_space."""
    config = copy.deepcopy(base_config)

    learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True)
    batch_size = trial.suggest_categorical("train_batch_size", [8, 16, 32])
    max_length = trial.suggest_categorical("max_length", [128, 256])
    weight_decay = trial.suggest_float("weight_decay", 0.0, 0.10)
    epochs = trial.suggest_int("epochs", 3, 10)
    constraint_lambda = trial.suggest_float("constraint_lambda", 0.05, 0.70)

    config["training"]["learning_rate"] = learning_rate
    config["training"]["train_batch_size"] = batch_size
    config["training"]["weight_decay"] = weight_decay
    config["training"]["epochs"] = epochs
    config["model"]["max_length"] = max_length
    config.setdefault("neuro_symbolic", {})["constraint_lambda"] = constraint_lambda

    base_out = Path(config["paths"]["output_dir"])
    trial_out = base_out / f"trial_{trial.number}"
    config["paths"]["output_dir"] = str(trial_out)

    print(f"\n{'='*50}")
    print(f"BẮT ĐẦU TRIAL {trial.number}")
    print(
        f"LR: {learning_rate:.2e}, Batch: {batch_size}, MaxLen: {max_length}, "
        f"WD: {weight_decay:.4f}, Epochs: {epochs}, λ: {constraint_lambda:.4f}"
    )
    print(f"{'='*50}")
    
    try:
        run_results = train_once(config)
        trainer = run_results["trainer"]
        tokenizer = run_results["tokenizer"]
        labels = run_results["labels"]
        df_val = run_results["df_val"]
        
        max_len = config["model"]["max_length"]
        val_dataset = TextDataset(df_val, tokenizer, max_len)
        _, _, metrics = evaluate_split(trainer, val_dataset, labels)
        
        macro_f1 = metrics["macro_f1"]
        print(f"Trial {trial.number} hoàn tất. Macro-F1: {macro_f1:.4f}")
        return macro_f1
        
    except Exception as e:
        print(f"Trial {trial.number} thất bại: {e}")
        raise optuna.TrialPruned()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/train.yml")
    parser.add_argument("--task", type=str, default="topic", choices=["topic", "action"])
    parser.add_argument("--trials", type=int, default=10)
    args = parser.parse_args()
    
    raw_config = load_yaml_config(Path(args.config))
    base_config = resolve_runtime_config(raw_config, task=args.task)
    
    base_config.setdefault("neuro_symbolic", {})["enabled"] = True
    
    study_name = f"esg_{args.task}_hyperopt"
    storage_name = f"sqlite:///{study_name}.db"
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction="maximize",
        load_if_exists=True
    )
    
    print(f"Bắt đầu Optuna tuning cho task: {args.task} với {args.trials} trials.")
    study.optimize(lambda trial: objective(trial, base_config), n_trials=args.trials)
    
    print("\n" + "="*50)
    print("HOÀN TẤT TUNING TỐI ƯU")
    print("="*50)
    print("Best Trial:")
    trial = study.best_trial
    print(f"  Value (Macro F1): {trial.value:.4f}")
    print("  Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
        
    out_path = Path(base_config["paths"]["output_dir"]) / f"best_params_{args.task}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(trial.params, f, indent=2)
    print(f"Đã lưu best parameters vào {out_path}")

if __name__ == "__main__":
    main()
