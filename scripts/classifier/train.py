"""
scripts/classifier/train.py

Fine-tunes CodeBERT for binary AI-generated code detection.
Outputs a model that scores P(AI-generated) for each code snippet.

Usage:
    python scripts/classifier/train.py

Checkpoints and final model saved to: data/models/
"""

import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

BASE      = Path(__file__).parent.parent.parent
DATA_DIR  = BASE / "data/dataset"
MODEL_DIR = BASE / "data/models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

CFG = {
    "model_name":        "microsoft/codebert-base",
    "max_length":        512,
    "batch_size":        16,
    "grad_accum_steps":  2,       # effective batch = 32
    "learning_rate":     2e-5,
    "weight_decay":      0.01,
    "epochs":            5,
    "warmup_ratio":      0.1,
    "early_stop_patience": 2,     # stop if val loss doesn't improve
    "dropout":           0.1,
    "seed":              42,
}


# ── Device ────────────────────────────────────────────────────────────────────

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple MPS (M-series GPU)")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    return device


# ── Dataset ───────────────────────────────────────────────────────────────────

class CodeDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int):
        self.codes  = df["code"].tolist()
        self.labels = df["label"].tolist()
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.codes)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.codes[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":          torch.tensor(self.labels[idx], dtype=torch.float),
        }


# ── Model ─────────────────────────────────────────────────────────────────────

class CodeBERTClassifier(nn.Module):
    def __init__(self, model_name: str, dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size  = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # Use [CLS] token representation
        cls = outputs.last_hidden_state[:, 0, :]
        cls = self.dropout(cls)
        logits = self.classifier(cls).squeeze(-1)
        return logits


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    all_preds  = []
    all_probs  = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].to(device)

            logits = model(input_ids, attention_mask)
            loss   = criterion(logits, labels)
            total_loss += loss.item()

            probs  = torch.sigmoid(logits).cpu().numpy()
            preds  = (probs >= 0.5).astype(int)
            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    avg_loss  = total_loss / len(loader)
    acc       = accuracy_score(all_labels, all_preds)
    f1        = f1_score(all_labels, all_preds, zero_division=0)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall    = recall_score(all_labels, all_preds, zero_division=0)
    auroc     = roc_auc_score(all_labels, all_probs)

    return {
        "loss":      avg_loss,
        "accuracy":  acc,
        "f1":        f1,
        "precision": precision,
        "recall":    recall,
        "auroc":     auroc,
    }


# ── Per-language evaluation ───────────────────────────────────────────────────

def evaluate_per_language(model, df_test, tokenizer, device):
    model.eval()
    results = {}
    for lang in df_test["language"].unique():
        subset = df_test[df_test["language"] == lang].copy()
        ds     = CodeDataset(subset, tokenizer, CFG["max_length"])
        loader = DataLoader(ds, batch_size=CFG["batch_size"] * 2, shuffle=False)

        probs_all  = []
        labels_all = []
        with torch.no_grad():
            for batch in loader:
                logits = model(
                    batch["input_ids"].to(device),
                    batch["attention_mask"].to(device),
                )
                probs = torch.sigmoid(logits).cpu().numpy()
                probs_all.extend(probs.tolist())
                labels_all.extend(batch["label"].numpy().tolist())

        preds = (np.array(probs_all) >= 0.5).astype(int)
        results[lang] = {
            "n":         len(labels_all),
            "accuracy":  accuracy_score(labels_all, preds),
            "f1":        f1_score(labels_all, preds, zero_division=0),
            "auroc":     roc_auc_score(labels_all, probs_all),
        }
    return results


# ── Training ──────────────────────────────────────────────────────────────────

def train():
    torch.manual_seed(CFG["seed"])
    np.random.seed(CFG["seed"])
    device = get_device()

    # Load data
    logger.info("Loading dataset...")
    df_train = pd.read_csv(DATA_DIR / "train.csv")
    df_val   = pd.read_csv(DATA_DIR / "val.csv")
    df_test  = pd.read_csv(DATA_DIR / "test.csv")

    # Drop rows with empty code
    for df in [df_train, df_val, df_test]:
        df.dropna(subset=["code"], inplace=True)
        df = df[df["code"].str.strip() != ""]

    logger.info(f"Train: {len(df_train):,}  Val: {len(df_val):,}  Test: {len(df_test):,}")

    # Class weights for mild imbalance
    n_human = (df_train["label"] == 0).sum()
    n_ai    = (df_train["label"] == 1).sum()
    pos_weight = torch.tensor([n_human / n_ai], dtype=torch.float).to(device)
    logger.info(f"Class balance — Human: {n_human:,}  AI: {n_ai:,}  pos_weight: {pos_weight.item():.3f}")

    # Tokenizer
    logger.info(f"Loading tokenizer: {CFG['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(CFG["model_name"])

    # Datasets and loaders
    train_ds = CodeDataset(df_train, tokenizer, CFG["max_length"])
    val_ds   = CodeDataset(df_val,   tokenizer, CFG["max_length"])

    train_loader = DataLoader(
        train_ds, batch_size=CFG["batch_size"],
        shuffle=True, num_workers=0, pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=CFG["batch_size"] * 2,
        shuffle=False, num_workers=0,
    )

    # Model
    logger.info(f"Loading model: {CFG['model_name']}")
    model = CodeBERTClassifier(CFG["model_name"], CFG["dropout"]).to(device)

    # Optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=CFG["learning_rate"],
        weight_decay=CFG["weight_decay"],
    )
    total_steps  = (len(train_loader) // CFG["grad_accum_steps"]) * CFG["epochs"]
    warmup_steps = int(total_steps * CFG["warmup_ratio"])
    scheduler    = get_linear_schedule_with_warmup(
        optimizer, warmup_steps, total_steps
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    logger.info(f"Total steps: {total_steps}  Warmup: {warmup_steps}")

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss    = float("inf")
    patience_counter = 0
    history          = []

    for epoch in range(1, CFG["epochs"] + 1):
        model.train()
        epoch_loss = 0.0
        step       = 0
        t0         = time.time()

        optimizer.zero_grad()
        for batch_idx, batch in enumerate(train_loader):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].to(device)

            logits = model(input_ids, attention_mask)
            loss   = criterion(logits, labels) / CFG["grad_accum_steps"]
            loss.backward()
            epoch_loss += loss.item() * CFG["grad_accum_steps"]

            if (batch_idx + 1) % CFG["grad_accum_steps"] == 0:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1

                if step % 100 == 0:
                    elapsed = time.time() - t0
                    logger.info(
                        f"  Epoch {epoch} step {step}/{total_steps // CFG['epochs']} "
                        f"loss={epoch_loss / (batch_idx + 1):.4f} "
                        f"elapsed={elapsed:.0f}s"
                    )

        # Validation
        val_metrics = evaluate(model, val_loader, device, criterion)
        train_loss  = epoch_loss / len(train_loader)
        elapsed     = time.time() - t0

        logger.info(
            f"Epoch {epoch}/{CFG['epochs']} — "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_metrics['loss']:.4f}  "
            f"val_f1={val_metrics['f1']:.4f}  "
            f"val_auroc={val_metrics['auroc']:.4f}  "
            f"({elapsed:.0f}s)"
        )

        history.append({"epoch": epoch, "train_loss": train_loss, **val_metrics})

        # Save checkpoint
        ckpt_path = MODEL_DIR / f"checkpoint_epoch{epoch}.pt"
        torch.save({
            "epoch":       epoch,
            "model_state": model.state_dict(),
            "optimizer":   optimizer.state_dict(),
            "val_metrics": val_metrics,
            "cfg":         CFG,
        }, ckpt_path)
        logger.info(f"  Checkpoint saved: {ckpt_path}")

        # Best model
        if val_metrics["loss"] < best_val_loss:
            best_val_loss    = val_metrics["loss"]
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_DIR / "best_model.pt")
            logger.info(f"  New best model saved (val_loss={best_val_loss:.4f})")
        else:
            patience_counter += 1
            logger.info(f"  No improvement ({patience_counter}/{CFG['early_stop_patience']})")
            if patience_counter >= CFG["early_stop_patience"]:
                logger.info("Early stopping triggered.")
                break

    # ── Final evaluation on test set ──────────────────────────────────────────
    logger.info("\nLoading best model for test evaluation...")
    model.load_state_dict(torch.load(MODEL_DIR / "best_model.pt", map_location=device))

    test_ds     = CodeDataset(df_test, tokenizer, CFG["max_length"])
    test_loader = DataLoader(test_ds, batch_size=CFG["batch_size"] * 2, shuffle=False)
    test_metrics = evaluate(model, test_loader, device, criterion)

    logger.info("\n" + "=" * 60)
    logger.info("TEST SET RESULTS")
    logger.info("=" * 60)
    for k, v in test_metrics.items():
        logger.info(f"  {k:<12} {v:.4f}")

    # Per-language breakdown
    logger.info("\nPER-LANGUAGE RESULTS:")
    per_lang = evaluate_per_language(model, df_test, tokenizer, device)
    for lang, metrics in sorted(per_lang.items()):
        logger.info(
            f"  {lang:<15} n={metrics['n']:>6}  "
            f"acc={metrics['accuracy']:.4f}  "
            f"f1={metrics['f1']:.4f}  "
            f"auroc={metrics['auroc']:.4f}"
        )

    # Save results
    results = {
        "cfg":          CFG,
        "history":      history,
        "test_metrics": test_metrics,
        "per_language": per_lang,
    }
    with open(MODEL_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {MODEL_DIR / 'results.json'}")

    # Save tokenizer alongside model for inference
    tokenizer.save_pretrained(MODEL_DIR / "tokenizer")
    logger.info(f"Tokenizer saved to {MODEL_DIR / 'tokenizer'}")

    # Save config
    with open(MODEL_DIR / "config.json", "w") as f:
        json.dump(CFG, f, indent=2)

    logger.info(f"\nDone. All artifacts in {MODEL_DIR}")


if __name__ == "__main__":
    train()