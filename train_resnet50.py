import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import os

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 4
EPOCHS = 10

TRAIN_DIR = "data"


def train():
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        zoom_range=0.1,
        horizontal_flip=True,
        validation_split=0.2
    )

    train_generator = datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )

    val_generator = datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )

    # 4. Build the model
    base_model = ResNet50(weights='imagenet', include_top=False,
                          input_shape=(224, 224, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # Freeze base model layers
    for layer in base_model.layers:
        layer.trainable = False

    # 5. Compile and train
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(train_generator, validation_data=val_generator, epochs=EPOCHS)

    # 6. Fine-tune top layers (optional)
    for layer in base_model.layers[-10:]:
        layer.trainable = True

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_generator, validation_data=val_generator, epochs=5)

    # 7. Save model
    model.save("resnet50_trash_classifier.h5")

    # 8. Convert to TFLite (optimized for Raspberry Pi + LiteRT)

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

    convert_to_tflite("resnet50_trash_classifier.h5",
                      "resnet50_trash_classifier.tflite")
