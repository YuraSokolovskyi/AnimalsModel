import numpy as np
import tensorflow as tf


def load_model(pathToModel):
    model = tf.keras.models.load_model(pathToModel)
    print("model loaded")
    return model


def modelPrediction(filePath, model):
    image = tf.keras.preprocessing.image.load_img(filePath, target_size=(150, 150), color_mode="grayscale")
    x = tf.keras.preprocessing.image.img_to_array(image)
    x = np.expand_dims(x, axis=0)
    print("image expanded")
    print("prediction: ")
    classes = model.predict(x)

    print(classes)

    if classes[0][0] == 1:
        return "Cat"
    elif classes[0][1] == 1:
        return "Horse"
    else:
        return "Panda"
