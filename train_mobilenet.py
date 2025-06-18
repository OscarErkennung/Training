import tensorflow as tf
from tensorflow.keras import layers, models
import os

# === Configuration ===
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
DATASET_DIR = "data1"


def convert_to_tflite(saved_model_path, output_path):
    converter = tf.lite.TFLiteConverter.from_keras_model(
        tf.keras.models.load_model(saved_model_path))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    def representative_dataset_gen():
        for _ in range(100):
            data, _ = next(train_generator)
            yield [data.astype("float32")]

    converter.representative_dataset = representative_dataset_gen
    tflite_model = converter.convert()

    with open(output_path, "wb") as f:
        f.write(tflite_model)


def train():
    # === Load and Augment Data ===
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE)

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE)

    # Prefetch for performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

    # === Data Augmentation Layer ===
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip('horizontal'),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomBrightness(0.1)
    ])

    # === Create Model ===
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SIZE + (3,),
                                                   include_top=False,
                                                   weights='imagenet')
    base_model.trainable = False  # freeze during initial training

    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    x = data_augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs, outputs)

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # === Train Initial Model ===
    initial_epochs = 5
    history = model.fit(train_ds, validation_data=val_ds,
                        epochs=initial_epochs)

    # === Fine-Tune Some Layers ===
    base_model.trainable = True
    fine_tune_at = 100  # freeze first 100 layers

    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),  # Lower LR for fine-tuning
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    fine_tune_epochs = 5
    total_epochs = initial_epochs + fine_tune_epochs

    model.fit(train_ds,
              validation_data=val_ds,
              epochs=total_epochs,
              initial_epoch=history.epoch[-1])

    # === Save Model ===
    model.save("trash_classifier_model")

    # === Optional: Convert to TensorFlow Lite ===
    converter = tf.lite.TFLiteConverter.from_saved_model(
        "trash_classifier_model")
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    with open("trash_classifier.tflite", "wb") as f:
        f.write(tflite_model)


if __name__ == "__main__":
    train()
