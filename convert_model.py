import tensorflow as tf

model = tf.keras.models.load_model("artifacts/lenet.keras")
print("Input shape:", model.input_shape)

model.save("artifacts/lenet.h5")
print("Modelo convertido a artifacts/lenet.h5")