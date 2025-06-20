from collections import Counter
import os
import tensorflow as tf
import datetime as dt

DATA_PATH = "data"
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
IMG_SHAPE = IMG_SIZE + (3,)

INITIAL_EPOCHS = 10
FINE_TUNING_EPOCHS = 7
FINE_TUNING_FROZEN_LAYERS = 80


train_dataset = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATA_PATH, "train"),
    shuffle=True,
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
)
validation_dataset = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATA_PATH, "val"),
    shuffle=True,
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
)


print(f"class names: {train_dataset.class_names}")


AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)

data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.2),
    ]
)

base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SHAPE, include_top=False, weights="imagenet"
)
base_model.trainable = False

image_batch, label_batch = next(iter(train_dataset))
feature_batch = base_model(image_batch)

# build pipeline

inputs = tf.keras.Input(shape=IMG_SHAPE)
x = data_augmentation(inputs)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
model = tf.keras.Model(inputs, outputs)


# initial training

print(f"starting initial training ({dt.datetime.now()})")

base_learning_rate = 0.0001
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.5, name="accuracy")],
)

history = model.fit(train_dataset, epochs=INITIAL_EPOCHS,
                    validation_data=validation_dataset)

# fine tuning

print(f"starting fine tuning ({dt.datetime.now()})")
print(f"the base model has {len(base_model.layers)} layers\n"
      f"freezing {FINE_TUNING_FROZEN_LAYERS} layers")


for layer in base_model.layers[:FINE_TUNING_FROZEN_LAYERS]:
    layer.trainable = False

model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.RMSprop(
        learning_rate=base_learning_rate / 10),
    metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.5, name="accuracy")],
)

history = model.fit(train_dataset, epochs=FINE_TUNING_EPOCHS,
                    validation_data=validation_dataset)

model_filename = (
    f"models/mobile_net_v2_{dt.datetime.now().strftime('%Y-%m-%d_%H:%M:%S.%f')}"
)
model.save(model_filename + ".keras")
print(f"model saved to {model_filename}")
