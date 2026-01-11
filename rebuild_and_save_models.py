"""
Rebuild models (CNN, GRU, Hybrid) from architecture used in phase3, load weights (*.weights.h5),
and save each model in TensorFlow SavedModel format under models/saved_*/.

Adjust shapes if needed to match training. This script prints progress and errors.
"""
import os
import traceback
import joblib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.optimizers import Adam

print("TensorFlow:", tf.__version__)

models_dir = 'models'

sequence_length = 12
n_features = 28  # from build config seen in notebook

# --- CNN ---
def build_cnn_model(sequence_length, n_features, learning_rate=0.0003):
    model = models.Sequential([
        layers.Input(shape=(sequence_length, n_features)),
        layers.Conv1D(64, 3, padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.002)),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Dropout(0.4),

        layers.Conv1D(128, 3, padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.002)),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Dropout(0.4),

        layers.Conv1D(256, 3, padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.002)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),

        layers.GlobalMaxPooling1D(),
        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.002)),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.002)),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.3),
        layers.Dense(3, activation='softmax')
    ])
    optimizer = Adam(learning_rate=learning_rate, clipnorm=1.0)
    # compile with a placeholder loss; we'll not train here
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# --- GRU ---
def build_gru_model(sequence_length, n_features, learning_rate=0.0003):
    model = models.Sequential([
        layers.Input(shape=(sequence_length, n_features)),
        layers.Bidirectional(layers.GRU(96, return_sequences=True,
                                       kernel_regularizer=regularizers.l2(0.003),
                                       recurrent_regularizer=regularizers.l2(0.003),
                                       dropout=0.3, recurrent_dropout=0.3)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),

        layers.Bidirectional(layers.GRU(64, return_sequences=True,
                                       kernel_regularizer=regularizers.l2(0.003),
                                       recurrent_regularizer=regularizers.l2(0.003),
                                       dropout=0.3, recurrent_dropout=0.3)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),

        layers.Bidirectional(layers.GRU(32, return_sequences=False,
                                       kernel_regularizer=regularizers.l2(0.002),
                                       recurrent_regularizer=regularizers.l2(0.002),
                                       dropout=0.2, recurrent_dropout=0.2)),
        layers.BatchNormalization(),
        layers.Dropout(0.4),

        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.002)),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.3),
        layers.Dense(3, activation='softmax')
    ])
    optimizer = Adam(learning_rate=learning_rate, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# --- Hybrid CNN-GRU (functional) ---
def build_hybrid_cnn_gru_model(sequence_length, n_temporal, n_static, learning_rate=0.0003):
    # temporal branch
    temp_in = layers.Input(shape=(sequence_length, n_temporal), name='temporal_input')
    x = layers.Conv1D(64, 3, padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.002))(temp_in)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Conv1D(128, 3, padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.002))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Conv1D(256, 3, padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.002))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Bidirectional(layers.GRU(64, return_sequences=False))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)

    # static branch
    stat_in = layers.Input(shape=(n_static,), name='static_input')
    s = layers.Dense(96, activation='relu', kernel_regularizer=regularizers.l2(0.003))(stat_in)
    s = layers.Dropout(0.4)(s)
    s = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.002))(s)
    s = layers.BatchNormalization()(s)
    s = layers.Dropout(0.3)(s)

    # fuse
    merged = layers.concatenate([x, s])
    merged = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.002))(merged)
    merged = layers.BatchNormalization()(merged)
    merged = layers.Dropout(0.4)(merged)
    merged = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001))(merged)
    merged = layers.Dropout(0.3)(merged)
    out = layers.Dense(3, activation='softmax')(merged)

    model = models.Model(inputs=[temp_in, stat_in], outputs=out)
    optimizer = Adam(learning_rate=learning_rate, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def try_load_and_save(model, weights_path, save_dir):
    print(f"\nLoading weights from: {weights_path}")
    try:
        model.load_weights(weights_path)
        print("  ✓ weights loaded")
    except Exception as e:
        print("  ✗ failed to load weights:", e)
        traceback.print_exc()
        return False
    try:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # Save a Keras native file (.keras) built with the current environment
        save_path = save_dir + '.keras'
        if os.path.exists(save_path):
            print(f"  → Save path already exists, skipping save: {save_path}")
        else:
            model.save(save_path)
            print(f"  ✓ Keras model saved to: {save_path}")
        return True
    except Exception as e:
        print("  ✗ failed to save model:", e)
        traceback.print_exc()
        return False


if __name__ == '__main__':
    # CNN
    print("\n== CNN ==")
    cnn = build_cnn_model(sequence_length, n_features)
    print(cnn.summary())
    cnn_weights = os.path.join(models_dir, 'cnn_best.weights.h5')
    try_load_and_save(cnn, cnn_weights, os.path.join(models_dir, 'cnn_saved_model'))

    # GRU
    print("\n== GRU ==")
    gru = build_gru_model(sequence_length, n_features)
    print(gru.summary())
    gru_weights = os.path.join(models_dir, 'gru_best.weights.h5')
    try_load_and_save(gru, gru_weights, os.path.join(models_dir, 'gru_saved_model'))

    # Hybrid
    print("\n== Hybrid ==")
    # Set hybrid feature counts consistent with notebook: temporal=17, static=15
    hybrid_temp = 17
    hybrid_stat = 15
    hybrid = build_hybrid_cnn_gru_model(sequence_length, hybrid_temp, hybrid_stat)
    print(hybrid.summary())
    hybrid_weights = os.path.join(models_dir, 'hybrid_best.weights.h5')
    try_load_and_save(hybrid, hybrid_weights, os.path.join(models_dir, 'hybrid_saved_model'))

    print("\nDone.")
