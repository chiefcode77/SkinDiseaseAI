# backend/model/create_dummy_tf_model.py
# Small Keras model saved as SavedModel so the loader can pick it up.
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent
SAVED_MODEL_DIR = MODEL_DIR / "saved_model"

def build_and_save():
    import tensorflow as tf
    from tensorflow.keras import layers, models

    # simple conv model for testing: input 224x224x3 -> softmax(3)
    inputs = layers.Input(shape=(224,224,3))
    x = layers.Rescaling(1./255)(inputs)
    x = layers.Conv2D(8, 3, activation='relu')(x)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(16, 3, activation='relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(32, activation='relu')(x)
    outputs = layers.Dense(3, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy')

    # Create directory and save model
    SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(str(SAVED_MODEL_DIR))
    print("Saved dummy TF model to:", SAVED_MODEL_DIR)

if __name__ == "__main__":
    build_and_save()
