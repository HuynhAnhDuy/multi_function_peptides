import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Dense, Dropout, Conv1D, MaxPooling1D, Flatten, Input,
    LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D, Add
)
from tensorflow.keras.optimizers import Adam

# ====== Base models (CNN, Transformer) ======
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

def create_transformer(input_shape, embed_dim=128, num_heads=4, ff_dim=128):
    inputs = Input(shape=input_shape)
    x = Conv1D(embed_dim, kernel_size=1, activation='relu')(inputs)

    # Self-attention block
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(x, x)
    attn_output = Dropout(0.1)(attn_output)
    out1 = Add()([x, attn_output])
    out1 = LayerNormalization(epsilon=1e-6)(out1)

    # FFN block
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

# ====== Meta model (FCNN stacked on top of CNN + Transformer outputs) ======
def create_meta_model(input_dim):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# ====== Helper: load and reshape features ======
def load_and_reshape_features(feature_csv, max_length=None):
    """
    Đọc file onehot+esm, bỏ cột Sequence nếu có, reshape thành (N, L, 20).
    """
    df = pd.read_csv(feature_csv)

    if "Sequence" in df.columns:
        df = df.drop(columns=["Sequence"])

    X = df.values.astype(np.float32)

    if max_length is None:
        # suy ra từ vector dài D = L * 20
        max_length = X.shape[1] // 20

    X = X.reshape((-1, max_length, 20))
    return X, max_length

# ====== Pipeline cho từng endpoint ======
def run_endpoint(
    name,
    train_feat_csv,
    train_label_csv,
    test_feat_csv,
    out_dir,
    epochs_base=30,
    epochs_meta=50,
    batch_size=32
):
    print(f"\n========== Endpoint: {name} ==========")

    # ----- Load training data -----
    # X_train
    X_train, max_length = load_and_reshape_features(train_feat_csv, max_length=None)

    # y_train
    df_y = pd.read_csv(train_label_csv)
    if "Label" not in df_y.columns:
        raise ValueError(f"{train_label_csv} must contain column 'Label'")
    y_train = df_y["Label"].astype(int).values

    print(f"[INFO] X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

    # ----- Base models -----
    input_shape = X_train.shape[1:]  # (L, 20)

    cnn = create_cnn(input_shape)
    transformer = create_transformer(input_shape)

    # Train base models trên training set
    cnn.fit(X_train, y_train, epochs=epochs_base, batch_size=batch_size, verbose=0)
    transformer.fit(X_train, y_train, epochs=epochs_base, batch_size=batch_size, verbose=0)

    # Stack: lấy prediction của base models trên CHÍNH X_train để train meta model
    cnn_train_pred = cnn.predict(X_train, verbose=0).flatten()
    trf_train_pred = transformer.predict(X_train, verbose=0).flatten()

    meta_train_input = np.column_stack([cnn_train_pred, trf_train_pred])
    print(f"[INFO] Meta-train input shape: {meta_train_input.shape}")

    meta_model = create_meta_model(meta_train_input.shape[1])
    meta_model.fit(meta_train_input, y_train, epochs=epochs_meta, batch_size=batch_size, verbose=0)

    # ----- Predict cho candidate peptides (test_feat_csv) -----
    # Lấy cả features và sequence từ cùng một file
    df_test = pd.read_csv(test_feat_csv)
    if "Sequence" not in df_test.columns:
        raise ValueError(f"{test_feat_csv} must contain column 'Sequence'")
    sequences = df_test["Sequence"].astype(str).values

    X_test, _ = load_and_reshape_features(test_feat_csv, max_length=max_length)
    print(f"[INFO] X_test (candidates) shape: {X_test.shape}")

    if len(sequences) != X_test.shape[0]:
        raise ValueError(
            f"Number of sequences ({len(sequences)}) != number of feature rows ({X_test.shape[0]}) "
            f"for endpoint {name}. Kiểm tra lại file {test_feat_csv}."
        )

    cnn_test_pred = cnn.predict(X_test, verbose=0).flatten()
    trf_test_pred = transformer.predict(X_test, verbose=0).flatten()
    meta_test_input = np.column_stack([cnn_test_pred, trf_test_pred])

    final_prob = meta_model.predict(meta_test_input, verbose=0).flatten()
    predicted_label = (final_prob >= 0.5).astype(int)

    # ----- Lưu kết quả -----
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, f"{name}_candidate_probability.csv")

    df_result = pd.DataFrame({
        "sequence": sequences,
        "probability": final_prob,
        "predicted_label": predicted_label
    })

    df_result.to_csv(out_csv, index=False)
    print(f"[✓] Saved predictions for {name} to: {out_csv}")
    print(f"    N_candidate peptides: {len(df_result)}")


def main():
    # Folder chứa outputs
    out_dir = "ACP_candidate_probability"

    # Cấu hình cho 3 endpoint
    endpoints = [
        {
            "name": "Liver",
            "train_feat": "Liver_train_cd40_onehot_esm_candidate.csv",
            "train_label": "Liver_train_cd40_y.csv",
            "test_feat": "Liver_test_cd40_onehot_esm_candidate.csv",
        },
        {
            "name": "Lung",
            "train_feat": "Lung_train_cd40_onehot_esm_candidate.csv",
            "train_label": "Lung_train_cd40_y.csv",
            "test_feat": "Lung_test_cd40_onehot_esm_candidate.csv",
        },
        {
            "name": "Colon",
            "train_feat": "Colon_train_cd40_onehot_esm_candidate.csv",
            "train_label": "Colon_train_cd40_y.csv",
            "test_feat": "Colon_test_cd40_onehot_esm_candidate.csv",
        },
    ]

    for cfg in endpoints:
        run_endpoint(
            name=cfg["name"],
            train_feat_csv=cfg["train_feat"],
            train_label_csv=cfg["train_label"],
            test_feat_csv=cfg["test_feat"],
            out_dir=out_dir,
            epochs_base=30,
            epochs_meta=50,
            batch_size=32,
        )

    print("\n[✓] Finished predicting probabilities for all endpoints.")


if __name__ == "__main__":
    main()
