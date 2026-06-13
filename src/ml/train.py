"""
DistilBERT fine-tuning for emotion classification with class weighting.
Run: python -m src.ml.train
"""
import torch
from transformers import (
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
)
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import json
import os
from datetime import datetime

from .data import (
    load_emotion_dataset,
    get_tokenizer,
    preprocess_dataset,
    compute_class_weights,
    get_label_names,
    get_num_labels,
    prepare_dataset,
)

MODEL_OUTPUT_DIR = "models/distilbert-emotion-v1"
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)


def make_compute_metrics(label_names):
    """
    Factory function that creates a compute_metrics function with
    access to label_names via closure. Avoids global state.
    """
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = accuracy_score(labels, predictions)

        report = classification_report(
            labels,
            predictions,
            target_names=label_names,
            output_dict=True,
            zero_division=0,
        )

        return {
            'accuracy': accuracy,
            'macro_f1': report['macro avg']['f1-score'],
            'weighted_f1': report['weighted avg']['f1-score'],
            'recall_surprise': report.get('surprise', {}).get('recall', 0.0),
            'recall_love': report.get('love', {}).get('recall', 0.0),
        }
    return compute_metrics


class WeightedTrainer(Trainer):
    """Custom Trainer with class-weighted loss."""

    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Override loss with class-weighted cross-entropy."""
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        loss_fct = torch.nn.CrossEntropyLoss(
            weight=self.class_weights.to(logits.device)
        )
        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss


def save_model_info(metrics: dict, num_labels: int, label_names: list) -> None:
    """Save training metadata alongside model artifacts."""
    info = {
        "model_name": "distilbert-base-uncased",
        "dataset": "dair-ai/emotion",
        "num_labels": num_labels,
        "label_names": label_names,
        "training_date": datetime.now().isoformat(),
        "metrics": {
            k: float(v) if isinstance(v, (np.floating, np.integer)) else v
            for k, v in metrics.items()
        },
    }

    with open(os.path.join(MODEL_OUTPUT_DIR, "model_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    print(f"Model info saved to {MODEL_OUTPUT_DIR}/model_info.json")


def main():
    print("Loading dataset...")
    tokenized_dataset, tokenizer = prepare_dataset()
    dataset = load_emotion_dataset()

    train_size = len(tokenized_dataset['train'])
    val_size = len(tokenized_dataset['validation'])
    test_size = len(tokenized_dataset['test'])
    print(f"Train: {train_size}, Val: {val_size}, Test: {test_size}")

    print("Computing class weights...")
    class_weights = compute_class_weights(dataset)
    label_names = get_label_names(dataset)
    num_labels = get_num_labels(dataset)

    weight_info = dict(zip(label_names, [f"{w:.3f}" for w in class_weights.tolist()]))
    print(f"Class weights: {weight_info}")

    print("Loading model...")
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=num_labels,
        id2label={i: name for i, name in enumerate(label_names)},
        label2id={name: i for i, name in enumerate(label_names)},
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir="./checkpoints",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=100,
        report_to="none",
    )

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['validation'],
        data_collator=data_collator,
        compute_metrics=make_compute_metrics(label_names),  # Closure, no globals
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print("Starting training...")
    trainer.train()

    print("Evaluating on validation set...")
    eval_results = trainer.evaluate()
    print(f"Validation results: {eval_results}")

    print(f"Saving model to {MODEL_OUTPUT_DIR}...")
    model.save_pretrained(MODEL_OUTPUT_DIR)
    tokenizer.save_pretrained(MODEL_OUTPUT_DIR)
    save_model_info(eval_results, num_labels, label_names)

    print("Done! Model ready for inference.")


if __name__ == "__main__":
    main()