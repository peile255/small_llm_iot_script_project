from pathlib import Path
import argparse
import json
import random
import time

import numpy as np
import pandas as pd
import torch
import yaml

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    set_seed,
)

from datasets import Dataset


ENCODINGS = [
    "utf-8-sig",
    "utf-8",
    "utf-16",
    "utf-16-le",
    "cp1252",
    "latin-1",
]


def read_script(path):
    data = Path(path).read_bytes()

    for enc in ENCODINGS:
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue

    return data.decode(
        "latin-1",
        errors="replace",
    )


def load_manifest(csv_path):
    df = pd.read_csv(csv_path)

    texts = []

    for i, p in enumerate(
        df["path"].astype(str)
    ):
        path = Path(p)

        if not path.exists():
            raise FileNotFoundError(
                f"Missing script: {path}"
            )

        texts.append(
            read_script(path)
        )

        if (i + 1) % 500 == 0:
            print(
                f"loaded {i+1}/{len(df)}"
            )

    out = pd.DataFrame(
        {
            "text": texts,
            "label": (
                df["label"]
                .astype(int)
                .to_numpy()
            ),
        }
    )

    return df, out


def metrics_from_predictions(
    y_true,
    y_pred,
):
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    mcc = matthews_corrcoef(
        y_true,
        y_pred,
    )

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    fpr = (
        fp / (fp + tn)
        if (fp + tn)
        else np.nan
    )

    fnr = (
        fn / (fn + tp)
        if (fn + tp)
        else np.nan
    )

    return {
        "n": int(len(y_true)),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mcc": float(mcc),
        "fpr": float(fpr),
        "fnr": float(fnr),
    }


def compute_metrics(eval_pred):
    logits, labels = eval_pred

    preds = np.argmax(
        logits,
        axis=1,
    )

    return {
        "accuracy": accuracy_score(
            labels,
            preds,
        ),
        "precision": precision_score(
            labels,
            preds,
            zero_division=0,
        ),
        "recall": recall_score(
            labels,
            preds,
            zero_division=0,
        ),
        "f1": f1_score(
            labels,
            preds,
            zero_division=0,
        ),
        "mcc": matthews_corrcoef(
            labels,
            preds,
        ),
    }


def predict_dataset(
    trainer,
    dataset,
    manifest,
    split_name,
    outdir,
):
    print(
        f"\nPredicting {split_name}..."
    )

    start = time.time()

    result = trainer.predict(
        dataset
    )

    elapsed = (
        time.time()
        - start
    )

    logits = result.predictions

    probs = torch.softmax(
        torch.tensor(logits),
        dim=1,
    ).numpy()

    preds = np.argmax(
        logits,
        axis=1,
    )

    y_true = (
        dataset["label"]
        if "label" in dataset.column_names
        else None
    )

    y_true = np.asarray(
        y_true,
        dtype=int,
    )

    metrics = metrics_from_predictions(
        y_true,
        preds,
    )

    metrics[
        "prediction_elapsed_s"
    ] = float(elapsed)

    metrics[
        "throughput_files_s"
    ] = float(
        len(dataset) / elapsed
    )

    pred_df = manifest.copy()

    pred_df["y_true"] = y_true
    pred_df["y_pred_codebert"] = preds

    pred_df[
        "prob_benign_codebert"
    ] = probs[:, 0]

    pred_df[
        "prob_malicious_codebert"
    ] = probs[:, 1]

    pred_df.to_csv(
        outdir
        / f"{split_name}_predictions.csv",
        index=False,
    )

    with open(
        outdir
        / f"{split_name}_metrics.json",
        "w",
    ) as f:
        json.dump(
            metrics,
            f,
            indent=2,
        )

    print(
        split_name,
        metrics,
    )

    return metrics


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
    )

    args = parser.parse_args()

    with open(
        args.config,
        "r",
    ) as f:
        cfg = yaml.safe_load(f)

    outdir = Path(
        cfg["output_dir"]
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    seed = int(
        cfg["seed"]
    )

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )

    print(
        "torch:",
        torch.__version__,
    )

    print(
        "cuda available:",
        torch.cuda.is_available(),
    )

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    print(
        "\nLoading tokenizer/model:",
        cfg["model_name"],
    )

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            cfg["model_name"]
        )
    )

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            cfg["model_name"],
            num_labels=2,
        )
    )

    print(
        "\nLoading manifests/scripts..."
    )

    train_manifest, train_df = (
        load_manifest(
            cfg["train_csv"]
        )
    )

    val_manifest, val_df = (
        load_manifest(
            cfg["val_csv"]
        )
    )

    test_manifest, test_df = (
        load_manifest(
            cfg["test_csv"]
        )
    )

    ext_manifest, ext_df = (
        load_manifest(
            cfg["external_csv"]
        )
    )

    train_ds = Dataset.from_pandas(
        train_df,
        preserve_index=False,
    )

    val_ds = Dataset.from_pandas(
        val_df,
        preserve_index=False,
    )

    test_ds = Dataset.from_pandas(
        test_df,
        preserve_index=False,
    )

    ext_ds = Dataset.from_pandas(
        ext_df,
        preserve_index=False,
    )

    max_length = int(
        cfg["max_length"]
    )

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
        )

    print(
        "\nTokenizing..."
    )

    train_tok = train_ds.map(
        tokenize,
        batched=True,
        remove_columns=["text"],
    )

    val_tok = val_ds.map(
        tokenize,
        batched=True,
        remove_columns=["text"],
    )

    test_tok = test_ds.map(
        tokenize,
        batched=True,
        remove_columns=["text"],
    )

    ext_tok = ext_ds.map(
        tokenize,
        batched=True,
        remove_columns=["text"],
    )

    collator = DataCollatorWithPadding(
        tokenizer=tokenizer
    )

    training_args = TrainingArguments(
        output_dir=str(
            outdir / "checkpoints"
        ),

        seed=seed,

        data_seed=seed,

        learning_rate=float(
            cfg["learning_rate"]
        ),

        weight_decay=float(
            cfg["weight_decay"]
        ),

        num_train_epochs=float(
            cfg["num_train_epochs"]
        ),

        per_device_train_batch_size=int(
            cfg["train_batch_size"]
        ),

        per_device_eval_batch_size=int(
            cfg["eval_batch_size"]
        ),

        warmup_ratio=float(
            cfg["warmup_ratio"]
        ),

        evaluation_strategy="epoch",

        save_strategy="epoch",

        logging_strategy="steps",

        logging_steps=50,

        load_best_model_at_end=True,

        metric_for_best_model=(
            cfg[
                "metric_for_best_model"
            ]
        ),

        greater_is_better=bool(
            cfg[
                "greater_is_better"
            ]
        ),

        save_total_limit=2,

        report_to=[],

        fp16=(
            torch.cuda.is_available()
        ),

        dataloader_num_workers=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    print(
        "\nStarting training..."
    )

    train_start = time.time()

    train_result = trainer.train()

    train_elapsed = (
        time.time()
        - train_start
    )

    print(
        "\nTraining finished:",
        train_elapsed,
        "seconds",
    )

    trainer.save_model(
        outdir / "best_model"
    )

    tokenizer.save_pretrained(
        outdir / "best_model"
    )

    train_summary = {
        "train_elapsed_s":
            float(train_elapsed),
        "best_metric":
            (
                trainer.state.best_metric
            ),
        "best_model_checkpoint":
            (
                trainer.state
                .best_model_checkpoint
            ),
        "train_metrics":
            train_result.metrics,
    }

    with open(
        outdir
        / "training_summary.json",
        "w",
    ) as f:
        json.dump(
            train_summary,
            f,
            indent=2,
            default=str,
        )

    print(
        "\nEvaluating best checkpoint..."
    )

    val_metrics = predict_dataset(
        trainer,
        val_tok,
        val_manifest,
        "validation",
        outdir,
    )

    test_metrics = predict_dataset(
        trainer,
        test_tok,
        test_manifest,
        "internal_test",
        outdir,
    )

    ext_metrics = predict_dataset(
        trainer,
        ext_tok,
        ext_manifest,
        "external_2000",
        outdir,
    )

    final_summary = {
        "model_name":
            cfg["model_name"],
        "seed":
            seed,
        "max_length":
            max_length,
        "validation":
            val_metrics,
        "internal_test":
            test_metrics,
        "external_2000":
            ext_metrics,
    }

    with open(
        outdir
        / "final_summary.json",
        "w",
    ) as f:
        json.dump(
            final_summary,
            f,
            indent=2,
        )

    print(
        "\nDONE"
    )

    print(
        json.dumps(
            final_summary,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
