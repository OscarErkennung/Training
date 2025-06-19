import datetime as dt
import tensorflow as tf
from tensorflow.keras import layers, models

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
DATASET_DIR = "data1"
INITIAL_EPOCHS = 10
FINE_TUNE_EPOCHS = 10
FROZEN_LAYERS = 30


def train():
    # === Load and Augment Data ===
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

    # === Data Augmentation Layer ===
    data_augmentation = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            layers.RandomBrightness(0.3),
            layers.RandomContrast(0.2),
            layers.RandomTranslation(0.2, 0.2)
        ]
    )

    # === Create Model ===
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet"
    )
    base_model.trainable = False  # freeze during initial training

    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    x = data_augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs, outputs)

    model.compile(optimizer="adam", loss="binary_crossentropy",
                  metrics=["accuracy"])

    # === Train Initial Model ===
    early_stop_callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True)
    print(f"starting initial training at {dt.datetime.now()}...")
    model.fit(train_ds, validation_data=val_ds,
              epochs=INITIAL_EPOCHS, callbacks=[early_stop_callback])

    # === Fine-Tune the later layers ===
    print(f"starting fine tuning at {dt.datetime.now()}...")
    base_model.trainable = True
    for layer in base_model.layers[:FROZEN_LAYERS]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),  # Lower LR for fine-tuning
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(train_ds, validation_data=val_ds, epochs=FINE_TUNE_EPOCHS)

    model_filename = f"models/mobile_net_v2_{dt.datetime.now().strftime('%Y-%m-%d_%H:%M:%S.%f')}"
    model.save(model_filename + ".keras")
    print(f"model saved to {model_filename}")


if __name__ == "__main__":
    train()
