import os
import tensorflow as tf
import numpy as np

path = "MendeleyData//test//healthy//" # change directory depending on set

correct_data = {
    'apple scab': 0,
    'black rot': 0,
    'cedar apple rust' : 0,
    'healthy': 0
}
class_names = ['apple scab','black rot','cedar apple rust','healthy']

order = []
percentage = []
model = tf.keras.models.load_model('apple_classification_model.keras')

directory = os.listdir(path)
for file in directory:

    img = tf.keras.utils.load_img(
        path+file, target_size=(255, 255)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array)


    correct_data[class_names[np.argmax(predictions)]] +=1
    order.append(class_names[np.argmax(predictions)])
    percentage.append(round(max(predictions[0]),4))