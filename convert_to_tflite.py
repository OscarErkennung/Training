import sys
import tensorflow as tf

model_filename = sys.argv[1]
model = tf.keras.models.load_model(model_filename)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
# converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()
with open(model_filename[:-len(".keras")] + ".tflite", "wb") as f:
    f.write(tflite_model)
