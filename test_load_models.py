from tensorflow import keras

models = ['cnn_model.keras', 'gru_model.keras', 'hybrid_model.keras']
for m in models:
    path = f"models/{m}"
    try:
        model = keras.models.load_model(path, compile=False)
        print(f"Loaded: {path} -> OK")
    except Exception as e:
        print(f"Failed: {path}\n  {e}")
