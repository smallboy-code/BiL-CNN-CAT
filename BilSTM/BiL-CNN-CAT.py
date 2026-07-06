# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
import random
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    hamming_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    auc as sk_auc,
)
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (
    Add,
    AveragePooling1D,
    Bidirectional,
    Concatenate,
    Conv1D,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    Input,
    LSTM,
    Lambda,
    MaxPooling1D,
    MultiHeadAttention,
    Multiply,
    Softmax,
)
from tensorflow.keras.optimizers import Adam

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

DEFAULT_TARGET_COLUMNS = [
    "肾虚（肾气阴）证",
    "风湿证",
    "瘀痹证",
    "肝风证",
    "溺毒证",
]

DEFAULT_ID_COLUMNS = ["id", "ID", "Id", "姓名", "name", "Name"]

CONTINUOUS_NAME_CANDIDATES = {
    "age",
    "年龄",
    "sbp",
    "收缩压",
    "dbp",
    "舒张压",
    "upc",
    "尿蛋白",
    "尿蛋白含量",
    "egfr",
    "eGFR",
    "估算肾小球滤过率",
}

CATEGORICAL_NAME_CANDIDATES = {
    "drbcs",
    "DRBCs",
    "畸形红细胞",
    "尿畸形红细胞",
    "舌质",
    "舌苔",
    "舌下脉络",
    "脉",
}

BINARY_MAP = {
    "是": 1,
    "否": 0,
    "有": 1,
    "无": 0,
    "阳性": 1,
    "阴性": 0,
    "男": 1,
    "女": 0,
    "male": 1,
    "female": 0,
    "Male": 1,
    "Female": 0,
    "M": 1,
    "F": 0,
    "true": 1,
    "false": 0,
    "True": 1,
    "False": 0,
    "TRUE": 1,
    "FALSE": 0,
    "1": 1,
    "0": 0,
}


@dataclass
class RunConfig:
    data: str
    outdir: str
    target_columns: List[str]
    id_columns: List[str]
    seed: int = 42
    holdout_test_size: float = 0.20
    calib_size: float = 0.20
    n_splits: int = 5
    epochs: int = 500
    batch_size: int = 32
    learning_rate: float = 1e-4
    dropout: float = 0.30
    lstm_units: int = 64
    conv_filters: int = 64
    conv_kernel_size: int = 3
    conv_stride: int = 1
    max_pool_size: int = 2
    max_pool_stride: int = 2
    attention_heads: int = 4
    attention_key_dim: int = 16
    spatial_pool_size: int = 2
    early_stopping_patience: int = 30
    threshold_step: float = 0.01
    run_holdout: bool = True
    run_cv: bool = True


def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)



def ensure_outdir(outdir: str) -> Path:
    path = Path(outdir)
    path.mkdir(parents=True, exist_ok=True)
    return path



def make_one_hot_encoder() -> OneHotEncoder:
    """Create OneHotEncoder with compatibility across scikit-learn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)



def normalize_column_names(cols: Iterable[str]) -> Dict[str, str]:
    return {col: str(col).strip() for col in cols}



def map_binary_like_values(df: pd.DataFrame) -> pd.DataFrame:
    """Map common Chinese/English binary values to 0/1 while preserving other values."""
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object:
            stripped = out[col].astype(str).str.strip()
            non_missing = out[col].dropna().astype(str).str.strip().unique().tolist()
            if non_missing and set(non_missing).issubset(set(BINARY_MAP.keys())):
                out[col] = stripped.map(BINARY_MAP).astype(float)
    return out



def convert_targets(y_df: pd.DataFrame) -> np.ndarray:
    y_clean = y_df.copy()
    for col in y_clean.columns:
        if y_clean[col].dtype == object:
            y_clean[col] = y_clean[col].astype(str).str.strip().map(BINARY_MAP)
        y_clean[col] = pd.to_numeric(y_clean[col], errors="coerce")
    y_clean = y_clean.fillna(0).astype(int)
    y_clean = (y_clean > 0).astype(int)
    return y_clean.values.astype(np.float32)



def multilabel_combo_labels(y: np.ndarray, min_count: int = 2) -> Optional[np.ndarray]:
    """
    Create combination labels for approximate stratification in multi-label data.
    Rare label combinations are grouped into a common 'rare' stratum when possible.
    """
    labels = np.array(["_".join(row.astype(int).astype(str)) for row in y])
    counts = pd.Series(labels).value_counts()
    grouped = np.array([label if counts[label] >= min_count else "rare_combo" for label in labels])
    if len(np.unique(grouped)) < 2:
        return None
    return grouped



def safe_train_calib_split(
    X: pd.DataFrame,
    y: np.ndarray,
    calib_size: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    stratify = multilabel_combo_labels(y, min_count=2)
    try:
        return train_test_split(
            X,
            y,
            test_size=calib_size,
            random_state=seed,
            shuffle=True,
            stratify=stratify,
        )
    except ValueError:
        return train_test_split(
            X,
            y,
            test_size=calib_size,
            random_state=seed,
            shuffle=True,
            stratify=None,
        )



def get_cv_splitter(y: np.ndarray, n_splits: int, seed: int):
    combo = multilabel_combo_labels(y, min_count=n_splits)
    if combo is not None:
        counts = pd.Series(combo).value_counts()
        if counts.min() >= n_splits:
            return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed), combo
    return KFold(n_splits=n_splits, shuffle=True, random_state=seed), None

def infer_feature_groups(X_train: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    """
    Infer continuous, categorical, and binary/numeric-pass-through columns.

    Manuscript-aligned rules:
    - continuous variables are standardized using training-set mean/SD;
    - binary variables are not standardized;
    - DRBCs and other categorical fields are one-hot encoded.
    """
    X_mapped = map_binary_like_values(X_train)
    continuous_cols: List[str] = []
    categorical_cols: List[str] = []
    binary_numeric_cols: List[str] = []

    for col in X_mapped.columns:
        col_str = str(col).strip()
        lower = col_str.lower()
        series = X_mapped[col]
        numeric_series = pd.to_numeric(series, errors="coerce")
        non_missing = numeric_series.dropna()
        unique_vals = set(non_missing.unique().tolist())

        is_named_continuous = col_str in CONTINUOUS_NAME_CANDIDATES or lower in CONTINUOUS_NAME_CANDIDATES
        is_named_categorical = col_str in CATEGORICAL_NAME_CANDIDATES or lower in CATEGORICAL_NAME_CANDIDATES
        is_binary_numeric = len(unique_vals) > 0 and unique_vals.issubset({0, 1})
        is_object_like = X_train[col].dtype == object

        if is_named_continuous:
            continuous_cols.append(col)
        elif is_named_categorical or (is_object_like and not is_binary_numeric):
            categorical_cols.append(col)
        elif is_binary_numeric:
            binary_numeric_cols.append(col)
        else:
            # Numeric variables not explicitly recognized as continuous are treated as continuous
            # if they have more than two observed values; otherwise they are binary-like.
            if len(unique_vals) > 2:
                continuous_cols.append(col)
            else:
                binary_numeric_cols.append(col)

    return continuous_cols, categorical_cols, binary_numeric_cols



def build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    X_train_mapped = map_binary_like_values(X_train)
    continuous_cols, categorical_cols, binary_numeric_cols = infer_feature_groups(X_train_mapped)

    transformers = []
    if continuous_cols:
        transformers.append(
            (
                "continuous",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="mean")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                continuous_cols,
            )
        )
    if categorical_cols:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", make_one_hot_encoder()),
                    ]
                ),
                categorical_cols,
            )
        )
    if binary_numeric_cols:
        transformers.append(
            (
                "binary_numeric",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent"))]),
                binary_numeric_cols,
            )
        )

    if not transformers:
        raise ValueError("No usable feature columns were detected.")

    return ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)



def fit_transform_preprocessor(
    X_train: pd.DataFrame,
    X_calib: pd.DataFrame,
    X_eval: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, ColumnTransformer]:
    X_train_mapped = map_binary_like_values(X_train)
    X_calib_mapped = map_binary_like_values(X_calib)
    X_eval_mapped = map_binary_like_values(X_eval)

    preprocessor = build_preprocessor(X_train_mapped)
    X_train_arr = preprocessor.fit_transform(X_train_mapped)
    X_calib_arr = preprocessor.transform(X_calib_mapped)
    X_eval_arr = preprocessor.transform(X_eval_mapped)

    return (
        np.asarray(X_train_arr, dtype=np.float32),
        np.asarray(X_calib_arr, dtype=np.float32),
        np.asarray(X_eval_arr, dtype=np.float32),
        preprocessor,
    )


class WeightedBinaryCrossentropyWithLogits(tf.keras.losses.Loss):
    """Label-wise weighted BCE implemented on logits."""

    def __init__(self, pos_weights: Sequence[float], name: str = "weighted_bce_with_logits"):
        super().__init__(name=name)
        self.pos_weights = tf.constant(pos_weights, dtype=tf.float32)

    def call(self, y_true, logits):
        y_true = tf.cast(y_true, tf.float32)
        logits = tf.cast(logits, tf.float32)
        loss = tf.nn.weighted_cross_entropy_with_logits(
            labels=y_true,
            logits=logits,
            pos_weight=self.pos_weights,
        )
        return tf.reduce_mean(loss)



def compute_pos_weights(y_train: np.ndarray, max_weight: float = 50.0) -> np.ndarray:
    positives = y_train.sum(axis=0)
    negatives = y_train.shape[0] - positives
    weights = np.ones(y_train.shape[1], dtype=np.float32)
    valid = positives > 0
    weights[valid] = negatives[valid] / np.maximum(positives[valid], 1.0)
    weights = np.clip(weights, 1.0, max_weight)
    return weights.astype(np.float32)



def build_bilstm_cnn_cat_model(
    input_dim: int,
    num_classes: int,
    cfg: RunConfig,
    pos_weights: Sequence[float],
) -> Model:
    """
    Build manuscript-aligned BiLSTM-CNN-CAT.

    Important reproducibility details:
    - input is (feature_position, 1), not (1, all_features);
    - Keras default initializers are used;
    - no batch-normalization layer is used;
    - no learning-rate decay schedule is applied;
    - early stopping is handled by callback using validation loss.
    """
    inputs = Input(shape=(input_dim, 1), name="feature_indexed_input")

    # Two-layer BiLSTM branch for global dependencies across ordered feature positions.
    x = Bidirectional(
        LSTM(cfg.lstm_units, return_sequences=True),
        name="bilstm_1",
    )(inputs)
    x = Dropout(cfg.dropout, name="dropout_after_bilstm_1")(x)
    x = Bidirectional(
        LSTM(cfg.lstm_units, return_sequences=True),
        name="bilstm_2",
    )(x)
    x = Dropout(cfg.dropout, name="dropout_after_bilstm_2")(x)

    # 1D-CNN branch for local feature-interaction patterns.
    x = Conv1D(
        filters=cfg.conv_filters,
        kernel_size=cfg.conv_kernel_size,
        strides=cfg.conv_stride,
        padding="same",
        activation="relu",
        name="conv1d_local_feature_interaction",
    )(x)
    x = MaxPooling1D(
        pool_size=cfg.max_pool_size,
        strides=cfg.max_pool_stride,
        padding="same",
        name="maxpool1d_local_feature_interaction",
    )(x)
    x = Dropout(cfg.dropout, name="dropout_after_cnn")(x)

    # Tunable spatial pooling, as described in the CAT module.
    if cfg.spatial_pool_size > 1:
        spatial = AveragePooling1D(
            pool_size=cfg.spatial_pool_size,
            strides=1,
            padding="same",
            name="cat_spatial_pooling",
        )(x)
    else:
        spatial = x

    # Multi-head residual attention over feature positions.
    mha = MultiHeadAttention(
        num_heads=cfg.attention_heads,
        key_dim=cfg.attention_key_dim,
        dropout=cfg.dropout,
        name="cat_multi_head_attention",
    )(spatial, spatial)
    mha = Dropout(cfg.dropout, name="dropout_after_cat_mha")(mha)
    attended = Add(name="cat_residual_attention")([spatial, mha])

    # Global average pooling provides a stable global context.
    global_context = GlobalAveragePooling1D(name="cat_global_average_pooling")(attended)

    logits = []
    for k in range(num_classes):
        # Label-specific attention scores over feature positions.
        score = Dense(
            cfg.conv_filters,
            activation="tanh",
            name=f"cat_label_{k}_score_hidden",
        )(attended)
        score = Dense(1, activation=None, name=f"cat_label_{k}_score")(score)
        weights = Softmax(axis=1, name=f"cat_label_{k}_attention_weights")(score)
        weighted = Multiply(name=f"cat_label_{k}_weighted_features")([attended, weights])
        label_vector = Lambda(
            lambda t: tf.reduce_sum(t, axis=1),
            name=f"cat_label_{k}_weighted_sum",
        )(weighted)

        label_representation = Concatenate(name=f"cat_label_{k}_local_global_fusion")(
            [label_vector, global_context]
        )
        label_representation = Dense(
            cfg.conv_filters,
            activation="relu",
            name=f"cat_label_{k}_representation",
        )(label_representation)
        label_representation = Dropout(cfg.dropout, name=f"cat_label_{k}_dropout")(
            label_representation
        )
        logit = Dense(1, activation=None, name=f"logit_label_{k}")(label_representation)
        logits.append(logit)

    outputs = Concatenate(axis=1, name="logits")(logits)

    model = Model(inputs=inputs, outputs=outputs, name="BiLSTM_CNN_CAT")
    model.compile(
        optimizer=Adam(learning_rate=cfg.learning_rate),
        loss=WeightedBinaryCrossentropyWithLogits(pos_weights=pos_weights),
    )
    return model



def to_feature_sequence(X: np.ndarray) -> np.ndarray:
    return np.asarray(X, dtype=np.float32)[..., np.newaxis]



def sigmoid_np(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float32)
    return 1.0 / (1.0 + np.exp(-logits))



def predict_proba(model: Model, X: np.ndarray, batch_size: int = 256) -> np.ndarray:
    logits = model.predict(to_feature_sequence(X), batch_size=batch_size, verbose=0)
    return sigmoid_np(logits)


def calibrate_thresholds(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    step: float = 0.01,
) -> np.ndarray:
    """
    Select per-label thresholds by maximizing label-wise F1 on the calibration set.
    If several thresholds tie, the smallest threshold is selected, matching the manuscript.
    """
    thresholds = []
    grid = np.round(np.arange(step, 1.0, step), 6)

    for k in range(y_true.shape[1]):
        best_threshold = 0.5
        best_f1 = -1.0
        for threshold in grid:
            pred = (y_proba[:, k] >= threshold).astype(int)
            score = f1_score(y_true[:, k], pred, zero_division=0)
            if score > best_f1 + 1e-12:
                best_f1 = score
                best_threshold = float(threshold)
            # Tie-breaking: keep the already selected smaller threshold.
        thresholds.append(best_threshold)

    return np.array(thresholds, dtype=np.float32)



def safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        if len(np.unique(y_true)) < 2:
            return np.nan
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return np.nan



def safe_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        if len(np.unique(y_true)) < 2:
            return np.nan
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        return float(sk_auc(recall, precision))
    except ValueError:
        return np.nan



def evaluate_multilabel(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    thresholds: np.ndarray,
    target_columns: Sequence[str],
) -> Dict[str, float]:
    y_pred = (y_proba >= thresholds.reshape(1, -1)).astype(int)

    metrics: Dict[str, float] = {
        "subset_accuracy_exact_match": float(accuracy_score(y_true, y_pred)),
        "labelwise_micro_accuracy": float(accuracy_score(y_true.ravel(), y_pred.ravel())),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
        "micro_precision": float(precision_score(y_true, y_pred, average="micro", zero_division=0)),
        "micro_recall": float(recall_score(y_true, y_pred, average="micro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }

    label_aucs = []
    label_pr_aucs = []
    label_briers = []
    for k, label in enumerate(target_columns):
        label_auc = safe_roc_auc(y_true[:, k], y_proba[:, k])
        label_pr_auc = safe_pr_auc(y_true[:, k], y_proba[:, k])
        try:
            label_brier = float(brier_score_loss(y_true[:, k], y_proba[:, k]))
        except ValueError:
            label_brier = np.nan

        label_aucs.append(label_auc)
        label_pr_aucs.append(label_pr_auc)
        label_briers.append(label_brier)

        safe_name = f"label_{k}_{label}"
        metrics[f"{safe_name}_auc"] = label_auc
        metrics[f"{safe_name}_pr_auc"] = label_pr_auc
        metrics[f"{safe_name}_brier"] = label_brier
        metrics[f"{safe_name}_threshold"] = float(thresholds[k])
        metrics[f"{safe_name}_prevalence"] = float(np.mean(y_true[:, k]))

    metrics["macro_auc_mean_of_labels"] = float(np.nanmean(label_aucs))
    metrics["macro_pr_auc_mean_of_labels"] = float(np.nanmean(label_pr_aucs))
    metrics["mean_brier_score"] = float(np.nanmean(label_briers))
    metrics["micro_auc_flattened"] = safe_roc_auc(y_true.ravel(), y_proba.ravel())

    return metrics



def save_json(obj: Dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)



def train_one_split(
    X_train_raw: pd.DataFrame,
    y_train_full: np.ndarray,
    X_eval_raw: pd.DataFrame,
    y_eval: np.ndarray,
    cfg: RunConfig,
    split_name: str,
    outdir: Path,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Train on a development training subset, calibrate thresholds on an internal
    validation/calibration subset, and evaluate on a hold-out or outer-fold set.
    """
    set_global_seed(cfg.seed)
    tf.keras.backend.clear_session()

    X_subtrain_raw, X_calib_raw, y_subtrain, y_calib = safe_train_calib_split(
        X_train_raw,
        y_train_full,
        calib_size=cfg.calib_size,
        seed=cfg.seed,
    )

    X_subtrain, X_calib, X_eval, preprocessor = fit_transform_preprocessor(
        X_subtrain_raw,
        X_calib_raw,
        X_eval_raw,
    )

    pos_weights = compute_pos_weights(y_subtrain)
    model = build_bilstm_cnn_cat_model(
        input_dim=X_subtrain.shape[1],
        num_classes=y_subtrain.shape[1],
        cfg=cfg,
        pos_weights=pos_weights,
    )

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=cfg.early_stopping_patience,
            restore_best_weights=True,
            verbose=0,
        )
    ]

    start_time = time.time()
    history = model.fit(
        to_feature_sequence(X_subtrain),
        y_subtrain,
        validation_data=(to_feature_sequence(X_calib), y_calib),
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        callbacks=callbacks,
        verbose=0,
    )
    elapsed_seconds = time.time() - start_time

    calib_proba = predict_proba(model, X_calib)
    thresholds = calibrate_thresholds(y_calib, calib_proba, step=cfg.threshold_step)

    eval_proba = predict_proba(model, X_eval)
    metrics = evaluate_multilabel(y_eval, eval_proba, thresholds, cfg.target_columns)
    metrics.update(
        {
            "split_name": split_name,
            "n_subtrain": int(X_subtrain.shape[0]),
            "n_calibration": int(X_calib.shape[0]),
            "n_evaluation": int(X_eval.shape[0]),
            "processed_feature_dim": int(X_subtrain.shape[1]),
            "epochs_run": int(len(history.history.get("loss", []))),
            "best_validation_loss": float(np.min(history.history.get("val_loss", [np.nan]))),
            "training_time_seconds": float(elapsed_seconds),
        }
    )

    reproducibility = {
        "split_name": split_name,
        "input_representation": "feature-indexed sequence with shape (processed_feature_dim, 1)",
        "lstm_layers": 2,
        "lstm_units": cfg.lstm_units,
        "conv_filters": cfg.conv_filters,
        "conv_kernel_size": cfg.conv_kernel_size,
        "conv_stride": cfg.conv_stride,
        "max_pool_size": cfg.max_pool_size,
        "max_pool_stride": cfg.max_pool_stride,
        "dropout": cfg.dropout,
        "attention_heads": cfg.attention_heads,
        "attention_key_dim": cfg.attention_key_dim,
        "spatial_pool_size": cfg.spatial_pool_size,
        "learning_rate": cfg.learning_rate,
        "learning_rate_decay": "none",
        "batch_normalization": "not used",
        "initializers": "Keras defaults: Glorot uniform for dense/convolution kernels and orthogonal recurrent initializer for LSTM recurrent kernels unless changed by TensorFlow/Keras defaults",
        "loss": "weighted binary cross-entropy with logits",
        "pos_weights": [float(x) for x in pos_weights],
        "thresholds": [float(x) for x in thresholds],
        "threshold_selection": "per-label F1 maximization on calibration subset, grid step 0.01, smallest threshold retained in ties",
        "optimizer": "Adam",
        "early_stopping": f"validation loss with patience={cfg.early_stopping_patience}, restore_best_weights=True",
        "epochs_max": cfg.epochs,
        "batch_size": cfg.batch_size,
        "training_time_seconds": float(elapsed_seconds),
    }

    # Save split-level artifacts.
    split_dir = outdir / split_name
    split_dir.mkdir(exist_ok=True, parents=True)
    pd.DataFrame(history.history).to_csv(split_dir / "training_history.csv", index=False)
    pd.DataFrame(eval_proba, columns=cfg.target_columns).to_csv(split_dir / "evaluation_probabilities.csv", index=False)
    pd.DataFrame((eval_proba >= thresholds.reshape(1, -1)).astype(int), columns=cfg.target_columns).to_csv(
        split_dir / "evaluation_predictions.csv", index=False
    )
    pd.DataFrame(y_eval.astype(int), columns=cfg.target_columns).to_csv(split_dir / "evaluation_true_labels.csv", index=False)
    save_json(reproducibility, split_dir / "reproducibility_details.json")

    return metrics, reproducibility


# -----------------------------------------------------------------------------
# Main experiment routines
# -----------------------------------------------------------------------------



def load_dataset(cfg: RunConfig) -> Tuple[pd.DataFrame, np.ndarray]:
    path = Path(cfg.data)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path.resolve()}")

    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    elif path.suffix.lower() in {".csv"}:
        df = pd.read_csv(path)
    else:
        raise ValueError("Unsupported data format. Use .xlsx, .xls, or .csv")

    missing_targets = [col for col in cfg.target_columns if col not in df.columns]
    if missing_targets:
        raise ValueError(f"Target columns not found in data: {missing_targets}")

    id_cols = [col for col in cfg.id_columns if col in df.columns]
    feature_cols = [col for col in df.columns if col not in cfg.target_columns and col not in id_cols]
    if not feature_cols:
        raise ValueError("No feature columns found after excluding targets and ID columns.")

    X = df[feature_cols].copy()
    y = convert_targets(df[cfg.target_columns])

    print(f"Loaded data: {path}")
    print(f"Samples: {X.shape[0]}")
    print(f"Raw feature columns: {X.shape[1]}")
    print(f"Target columns: {cfg.target_columns}")
    print("Label prevalence:")
    for label, prevalence in zip(cfg.target_columns, y.mean(axis=0)):
        print(f"  {label}: {prevalence:.4f} ({int(y[:, cfg.target_columns.index(label)].sum())}/{len(y)})")

    return X, y



def run_holdout_experiment(X: pd.DataFrame, y: np.ndarray, cfg: RunConfig, outdir: Path) -> Dict[str, float]:
    print("\n" + "=" * 80)
    print("Independent hold-out evaluation")
    print("=" * 80)

    stratify = multilabel_combo_labels(y, min_count=2)
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=cfg.holdout_test_size,
            random_state=cfg.seed,
            shuffle=True,
            stratify=stratify,
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=cfg.holdout_test_size,
            random_state=cfg.seed,
            shuffle=True,
            stratify=None,
        )

    metrics, reproducibility = train_one_split(
        X_train,
        y_train,
        X_test,
        y_test,
        cfg,
        split_name="holdout_test",
        outdir=outdir,
    )

    pd.DataFrame([metrics]).to_csv(outdir / "holdout_metrics.csv", index=False)
    save_json(reproducibility, outdir / "holdout_reproducibility_details.json")

    print_metrics(metrics)
    return metrics



def run_cross_validation(X: pd.DataFrame, y: np.ndarray, cfg: RunConfig, outdir: Path) -> pd.DataFrame:
    print("\n" + "=" * 80)
    print(f"{cfg.n_splits}-fold cross-validation")
    print("=" * 80)

    splitter, combo = get_cv_splitter(y, cfg.n_splits, cfg.seed)
    if combo is None:
        split_iter = splitter.split(X)
        print("Using KFold because some multi-label combinations are too rare for StratifiedKFold.")
    else:
        split_iter = splitter.split(X, combo)
        print("Using StratifiedKFold based on grouped multi-label combinations.")

    fold_metrics = []
    all_reproducibility = []

    for fold_idx, (train_idx, eval_idx) in enumerate(split_iter, start=1):
        print(f"\nFold {fold_idx}/{cfg.n_splits}")
        X_train = X.iloc[train_idx].copy()
        y_train = y[train_idx]
        X_eval = X.iloc[eval_idx].copy()
        y_eval = y[eval_idx]

        metrics, reproducibility = train_one_split(
            X_train,
            y_train,
            X_eval,
            y_eval,
            cfg,
            split_name=f"cv_fold_{fold_idx}",
            outdir=outdir,
        )
        metrics["fold"] = fold_idx
        fold_metrics.append(metrics)
        all_reproducibility.append(reproducibility)
        print_metrics(metrics, compact=True)

    fold_df = pd.DataFrame(fold_metrics)
    fold_df.to_csv(outdir / "cv_fold_metrics.csv", index=False)

    numeric_cols = fold_df.select_dtypes(include=[np.number]).columns.tolist()
    summary_rows = []
    for col in numeric_cols:
        if col == "fold":
            continue
        summary_rows.append(
            {
                "metric": col,
                "mean": float(fold_df[col].mean()),
                "std": float(fold_df[col].std(ddof=1)),
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(outdir / "cv_summary_mean_std.csv", index=False)
    save_json({"folds": all_reproducibility}, outdir / "cv_reproducibility_details.json")

    print("\nCross-validation summary: mean ± SD")
    for _, row in summary_df.iterrows():
        if row["metric"] in {
            "subset_accuracy_exact_match",
            "labelwise_micro_accuracy",
            "micro_precision",
            "micro_recall",
            "micro_f1",
            "macro_precision",
            "macro_recall",
            "macro_f1",
            "macro_auc_mean_of_labels",
            "micro_auc_flattened",
            "training_time_seconds",
            "epochs_run",
        }:
            print(f"  {row['metric']:<32}: {row['mean']:.4f} ± {row['std']:.4f}")

    return fold_df



def print_metrics(metrics: Dict[str, float], compact: bool = False) -> None:
    keys = [
        "subset_accuracy_exact_match",
        "labelwise_micro_accuracy",
        "micro_precision",
        "micro_recall",
        "micro_f1",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "macro_auc_mean_of_labels",
        "micro_auc_flattened",
        "mean_brier_score",
        "epochs_run",
        "training_time_seconds",
    ]
    if compact:
        print(
            f"  Micro-F1={metrics.get('micro_f1', np.nan):.4f}, "
            f"Macro-F1={metrics.get('macro_f1', np.nan):.4f}, "
            f"Micro-AUC={metrics.get('micro_auc_flattened', np.nan):.4f}, "
            f"Epochs={metrics.get('epochs_run', np.nan):.0f}, "
            f"Time={metrics.get('training_time_seconds', np.nan):.1f}s"
        )
    else:
        for key in keys:
            if key in metrics:
                value = metrics[key]
                if isinstance(value, (float, np.floating)):
                    print(f"  {key:<32}: {value:.4f}")
                else:
                    print(f"  {key:<32}: {value}")



def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(description="BiLSTM-CNN-CAT for IgAN multi-label clinical pattern prediction")
    parser.add_argument("--data", type=str, default="宏500处理.xlsx", help="Path to Excel/CSV data file")
    parser.add_argument("--outdir", type=str, default="results_bil_cnn_cat", help="Output directory")
    parser.add_argument("--targets", type=str, nargs="*", default=DEFAULT_TARGET_COLUMNS, help="Target columns")
    parser.add_argument("--id-columns", type=str, nargs="*", default=DEFAULT_ID_COLUMNS, help="ID columns to exclude")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--holdout-test-size", type=float, default=0.20)
    parser.add_argument("--calib-size", type=float, default=0.20)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.30)
    parser.add_argument("--lstm-units", type=int, default=64)
    parser.add_argument("--conv-filters", type=int, default=64)
    parser.add_argument("--conv-kernel-size", type=int, default=3)
    parser.add_argument("--conv-stride", type=int, default=1)
    parser.add_argument("--max-pool-size", type=int, default=2)
    parser.add_argument("--max-pool-stride", type=int, default=2)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--attention-key-dim", type=int, default=16)
    parser.add_argument("--spatial-pool-size", type=int, default=2)
    parser.add_argument("--early-stopping-patience", type=int, default=30)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument("--no-holdout", action="store_true", help="Disable independent hold-out evaluation")
    parser.add_argument("--no-cv", action="store_true", help="Disable cross-validation")

    args = parser.parse_args()
    return RunConfig(
        data=args.data,
        outdir=args.outdir,
        target_columns=list(args.targets),
        id_columns=list(args.id_columns),
        seed=args.seed,
        holdout_test_size=args.holdout_test_size,
        calib_size=args.calib_size,
        n_splits=args.n_splits,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        dropout=args.dropout,
        lstm_units=args.lstm_units,
        conv_filters=args.conv_filters,
        conv_kernel_size=args.conv_kernel_size,
        conv_stride=args.conv_stride,
        max_pool_size=args.max_pool_size,
        max_pool_stride=args.max_pool_stride,
        attention_heads=args.attention_heads,
        attention_key_dim=args.attention_key_dim,
        spatial_pool_size=args.spatial_pool_size,
        early_stopping_patience=args.early_stopping_patience,
        threshold_step=args.threshold_step,
        run_holdout=not args.no_holdout,
        run_cv=not args.no_cv,
    )



def main() -> None:
    cfg = parse_args()
    set_global_seed(cfg.seed)
    outdir = ensure_outdir(cfg.outdir)

    save_json(cfg.__dict__, outdir / "run_config.json")

    X, y = load_dataset(cfg)

    if cfg.run_holdout:
        run_holdout_experiment(X, y, cfg, outdir)

    if cfg.run_cv:
        run_cross_validation(X, y, cfg, outdir)

    print("\nDone.")
    print(f"Results saved to: {outdir.resolve()}")
    print("\nMulti-label reporting notes:")
    print("  - subset_accuracy_exact_match is strict exact-match accuracy;")
    print("  - labelwise_micro_accuracy is computed over all label-instance decisions;")
    print("  - micro precision/recall/F1 aggregate all labels globally;")
    print("  - macro precision/recall/F1 average label-wise metrics and are more sensitive to minority labels;")
    print("  - thresholds are calibrated per label on the internal validation/calibration subset only.")


if __name__ == "__main__":
    main()
