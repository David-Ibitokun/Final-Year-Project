"""
Build only the hybrid model, load hybrid_best.weights.h5, and save as models/hybrid_resaved.keras
"""
import os, traceback
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.optimizers import Adam

print('TensorFlow', tf.__version__)

sequence_length = 12
hybrid_temp = 17
hybrid_stat = 15

def build_hybrid_cnn_gru_model(sequence_length, n_temporal, n_static, learning_rate=0.0003):
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

    stat_in = layers.Input(shape=(n_static,), name='static_input')
    s = layers.Dense(96, activation='relu', kernel_regularizer=regularizers.l2(0.003))(stat_in)
    s = layers.Dropout(0.4)(s)
    s = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.002))(s)
    s = layers.BatchNormalization()(s)
    s = layers.Dropout(0.3)(s)

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

if __name__ == '__main__':
    try:
        model = build_hybrid_cnn_gru_model(sequence_length, hybrid_temp, hybrid_stat)
        print(model.summary())
        weights = os.path.join('models', 'hybrid_best.weights.h5')
        print('Loading weights from', weights)
        model.load_weights(weights)
        save_path = os.path.join('models', 'hybrid_resaved.keras')
        if os.path.exists(save_path):
            print('Save exists, skipping:', save_path)
        else:
            model.save(save_path)
            print('Saved to', save_path)
    except Exception as e:
        print('Error:', e)
        traceback.print_exc()
