import numpy as np
import pandas as pd
import tensorflow as tf
import random
import os

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Dense, Dropout, Conv1D, MaxPooling1D, Flatten, Input,
    LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D, Add
)

from sklearn.metrics import (
    accuracy_score, matthews_corrcoef, roc_auc_score, average_precision_score
)

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier


# ===============================
# Reproducibility
# ===============================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ===============================
# CNN Model
# ===============================

def create_cnn(input_shape):

    model = Sequential([
        Input(shape=input_shape),

        Conv1D(filters=64, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),

        Flatten(),

        Dense(100, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


# ===============================
# Transformer Model
# ===============================

def create_transformer(input_shape, embed_dim=128, num_heads=4, ff_dim=128):

    inputs = Input(shape=input_shape)

    x = Conv1D(embed_dim, kernel_size=1, activation='relu')(inputs)

    attn_output = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=embed_dim
    )(x, x)

    attn_output = Dropout(0.1)(attn_output)

    out1 = Add()([x, attn_output])
    out1 = LayerNormalization(epsilon=1e-6)(out1)

    ffn = Dense(ff_dim, activation='relu')(out1)
    ffn = Dense(embed_dim)(ffn)

    ffn = Dropout(0.1)(ffn)

    out2 = Add()([out1, ffn])
    out2 = LayerNormalization(epsilon=1e-6)(out2)

    x = GlobalAveragePooling1D()(out2)

    x = Dense(64, activation='relu')(x)

    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs, outputs)

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


# ===============================
# Evaluation Metrics
# ===============================

def evaluate_model(y_true, y_pred_prob):

    y_pred = (y_pred_prob > 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    auroc = roc_auc_score(y_true, y_pred_prob)
    auprc = average_precision_score(y_true, y_pred_prob)

    return acc, mcc, auroc, auprc


# ===============================
# Stacking Pipeline
# ===============================

def run_stacking(X_train, y_train, X_test, y_test,
                 n_splits=5,
                 n_repeats=3,
                 output_prefix="Stacking_RF"):

    os.makedirs("results", exist_ok=True)

    all_results = []

    for repeat in range(n_repeats):

        print(f"\n===== Training Run {repeat+1}/{n_repeats} =====")

        skf = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=SEED + repeat
        )

        oof_preds = np.zeros((X_train.shape[0], 2))
        test_preds = np.zeros((X_test.shape[0], 2))

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):

            print(f"Fold {fold+1}/{n_splits}")

            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]

            models = [
                create_cnn(X_train.shape[1:]),
                create_transformer(X_train.shape[1:])
            ]

            fold_test_preds = []

            for m_idx, model in enumerate(models):

                model.fit(
                    X_tr,
                    y_tr,
                    epochs=30,
                    batch_size=32,
                    verbose=0
                )

                val_pred = model.predict(X_val, verbose=0).flatten()
                oof_preds[val_idx, m_idx] = val_pred

                test_pred = model.predict(X_test, verbose=0).flatten()
                fold_test_preds.append(test_pred)

            test_preds += np.column_stack(fold_test_preds)

        test_preds /= n_splits

        # ===============================
        # Meta Model (Random Forest)
        # ===============================

        meta_model = RandomForestClassifier(
            n_estimators=300,
            random_state=SEED + repeat,
            n_jobs=-1
        )

        meta_model.fit(oof_preds, y_train)

        meta_y_pred_prob = meta_model.predict_proba(test_preds)[:, 1]

        acc, mcc, auroc, auprc = evaluate_model(y_test, meta_y_pred_prob)

        all_results.append({
            "Run": repeat + 1,
            "Accuracy": acc,
            "MCC": mcc,
            "AUROC": auroc,
            "AUPRC": auprc
        })

        print(
            f"→ ACC: {acc:.3f}, MCC: {mcc:.3f}, AUROC: {auroc:.3f}, AUPRC: {auprc:.3f}"
        )

    raw_df = pd.DataFrame(all_results)

    raw_path = f"results/{output_prefix}_raw.csv"
    raw_df.to_csv(raw_path, index=False)

    summary_df = pd.DataFrame({
        "Metric": ["Accuracy", "MCC", "AUROC", "AUPRC"],
        "Mean ± SD": [
            f"{raw_df['Accuracy'].mean():.3f} ± {raw_df['Accuracy'].std():.3f}",
            f"{raw_df['MCC'].mean():.3f} ± {raw_df['MCC'].std():.3f}",
            f"{raw_df['AUROC'].mean():.3f} ± {raw_df['AUROC'].std():.3f}",
            f"{raw_df['AUPRC'].mean():.3f} ± {raw_df['AUPRC'].std():.3f}",
        ]
    })

    summary_path = f"results/{output_prefix}_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\n✅ Saved results:")
    print(f" - Raw metrics: {raw_path}")
    print(f" - Summary: {summary_path}")

    return raw_df, summary_df


# ===============================
# Main
# ===============================

def main():

    X_train_df = pd.read_csv("Lung_train_cd40_onehot_esm.csv", index_col=0)
    y_train = pd.read_csv("Lung_train_cd40_y.csv")["Label"].values

    X_test_df = pd.read_csv("Lung_test_cd40_onehot_esm.csv", index_col=0)
    y_test = pd.read_csv("Lung_test_cd40_y.csv")["Label"].values

    for col in ["Sequence", "sequence"]:

        if col in X_train_df.columns:
            X_train_df = X_train_df.drop(columns=[col])

        if col in X_test_df.columns:
            X_test_df = X_test_df.drop(columns=[col])

    X_train_df = X_train_df.apply(pd.to_numeric, errors='coerce').fillna(0)
    X_test_df = X_test_df.apply(pd.to_numeric, errors='coerce').fillna(0)

    X_train = X_train_df.values.astype(np.float32)
    X_test = X_test_df.values.astype(np.float32)

    total_features = X_train.shape[1]

    assert total_features % 20 == 0, "Feature number must be divisible by 20."

    max_length = total_features // 20

    X_train = X_train.reshape((-1, max_length, 20))
    X_test = X_test.reshape((-1, max_length, 20))

    print("Input shapes:")
    print("X_train:", X_train.shape)
    print("X_test :", X_test.shape)

    run_stacking(
        X_train,
        y_train,
        X_test,
        y_test,
        n_repeats=3,
        output_prefix="Lung_Stacking_RF"
    )


if __name__ == "__main__":
    main()