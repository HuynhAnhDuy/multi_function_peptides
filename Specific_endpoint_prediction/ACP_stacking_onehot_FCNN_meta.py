import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Dense, Dropout, Conv1D, MaxPooling1D, Flatten, Input,
    LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D, Add
)
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import (
    accuracy_score, matthews_corrcoef, roc_auc_score, average_precision_score
)
import os


# ====== Define Base Models ======
def create_cnn(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(filters=64, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(100, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def create_transformer(input_shape, embed_dim=128, num_heads=4, ff_dim=128):
    inputs = Input(shape=input_shape)
    x = Conv1D(embed_dim, kernel_size=1, activation='relu')(inputs)

    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(x, x)
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
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# ====== Meta Model (FCNN) ======
def create_meta_model(input_shape):
    model = Sequential([
        Input(shape=(input_shape,)),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model


# ====== Evaluate 4 Metrics ======
def evaluate_model(y_true, y_pred_prob):
    y_pred = (y_pred_prob > 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    auroc = roc_auc_score(y_true, y_pred_prob)
    auprc = average_precision_score(y_true, y_pred_prob)
    return acc, mcc, auroc, auprc


# ====== Run Stacking ======
def run_stacking(X_train, y_train, X_test, y_test, n_repeats=3, output_prefix="Stacking_FCNN_onehot"):
    os.makedirs("results", exist_ok=True)
    all_results = []

    for repeat in range(n_repeats):
        print(f"\n===== Training Run {repeat+1}/{n_repeats} =====")

        # Train base models
        models = [
            create_cnn(X_train.shape[1:]),
            create_transformer(X_train.shape[1:])
        ]

        model_preds = []
        for model in models:
            model.fit(X_train, y_train, epochs=30, batch_size=32, verbose=0)
            model_preds.append(model.predict(X_test, verbose=0).flatten())

        # Meta model
        meta_X = np.column_stack(model_preds)
        meta_model = create_meta_model(meta_X.shape[1])
        meta_model.fit(meta_X, y_test, epochs=30, batch_size=32, verbose=0)

        meta_y_pred_prob = meta_model.predict(meta_X).flatten()
        acc, mcc, auroc, auprc = evaluate_model(y_test, meta_y_pred_prob)

        all_results.append({
            "Run": repeat + 1,
            "Accuracy": acc,
            "MCC": mcc,
            "AUROC": auroc,
            "AUPRC": auprc
        })

        print(f"→ ACC: {acc:.3f}, MCC: {mcc:.3f}, AUROC: {auroc:.3f}, AUPRC: {auprc:.3f}")

    # ===== Save raw results =====
    raw_df = pd.DataFrame(all_results)
    raw_path = f"results/{output_prefix}_raw.csv"
    raw_df.to_csv(raw_path, index=False)

    # ===== Save mean ± SD =====
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


# ====== Main ======
def main():
    # === Load One-hot data ===
    X_train_df = pd.read_csv("Liver_train_cd40_onehot_esm.csv", index_col=0)
    y_train = pd.read_csv("Liver_train_cd40_y.csv")["Label"].values
    X_test_df = pd.read_csv("Liver_test_cd40_onehot_esm.csv", index_col=0)
    y_test = pd.read_csv("Liver_test_cd40_y.csv")["Label"].values

    # === Remove 'Sequence' column if exists ===
    for col in ["Sequence", "sequence"]:
        if col in X_train_df.columns:
            X_train_df = X_train_df.drop(columns=[col])
        if col in X_test_df.columns:
            X_test_df = X_test_df.drop(columns=[col])

    # === Convert to float and fill NaN ===
    X_train_df = X_train_df.apply(pd.to_numeric, errors='coerce').fillna(0)
    X_test_df = X_test_df.apply(pd.to_numeric, errors='coerce').fillna(0)

    X_train = X_train_df.values.astype(np.float32)
    X_test = X_test_df.values.astype(np.float32)

    # === Check feature dimensions (20 features per amino acid) ===
    total_features = X_train.shape[1]
    assert total_features % 20 == 0, "⚠️ Total number of features is not divisible by 20 (check one-hot encoding)."
    max_length = total_features // 20

    # === Reshape into 3D tensor: [samples, seq_length, 20] ===
    X_train = X_train.reshape((-1, max_length, 20))
    X_test = X_test.reshape((-1, max_length, 20))

    print("✅ Input shapes (One-hot encoding):")
    print("  X_train:", X_train.shape)
    print("  X_test :", X_test.shape)

    # === Run stacking ===
    run_stacking(X_train, y_train, X_test, y_test, n_repeats=3, output_prefix="Liver_Stacking_FCNN_onehot_ESM")


if __name__ == "__main__":
    main()
