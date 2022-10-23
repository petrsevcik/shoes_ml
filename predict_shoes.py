from tensorflow import keras
from tensorflow.keras.preprocessing import image

import numpy as np
import matplotlib.pyplot as plt


import warnings
warnings.filterwarnings("ignore")

loaded_best_model = keras.models.load_model("model_06-0.69.h5")

def predict(img_rel_path):
    # Import Image from the path with size of (300, 300)
    img = image.load_img(img_rel_path, target_size=(300, 300))

    # Convert Image to a numpy array
    img = image.img_to_array(img, dtype=np.uint8)

    # Scaling the Image Array values between 0 and 1
    img = np.array(img) / 255.0

    # Plotting the Loaded Image
    plt.title("Loaded Image")
    plt.axis('off')
    plt.imshow(img.squeeze())
    plt.show()

    # Get the Predicted Label for the loaded Image
    p = loaded_best_model.predict(img[np.newaxis, ...])

    # Label array
    labels = {0: 'adidas', 1: 'altra', 2: 'asics', 3: 'joma', 4: 'new_balance', 5: 'nike'}

    print("\n\nMaximum Probability: ", np.max(p[0], axis=-1))
    predicted_class = labels[np.argmax(p[0], axis=-1)]
    print("Classified:", predicted_class, "\n\n")

    classes = []
    prob = []
    print("\n-------------------Individual Probability--------------------------------\n")

    for i, j in enumerate(p[0], 0):
        print(labels[i].upper(), ':', round(j * 100, 2), '%')
        classes.append(labels[i])
        prob.append(round(j * 100, 2))

    def plot_bar_x():
        # this is for plotting purpose
        index = np.arange(len(classes))
        plt.bar(index, prob)
        plt.xlabel('Labels', fontsize=8)
        plt.ylabel('Probability', fontsize=8)
        plt.xticks(index, classes, fontsize=8, rotation=20)
        plt.title('Probability for loaded image')
        plt.show()

    plot_bar_x()

# Images from same eshop (same as in training data)
# predict('test_model/Altra-Vanish-Carbon-Running-Shoes138584759.jpg') # not ok
# predict('test_model/Altra-Vanish-Tempo-Running-Shoes139063773_2.jpg') # ok
# predict('test_model/Asics-Gel-Cumulus-22-Running-Shoes137620907_3.jpg') # ok
#predict('test_model/adidas-Adizero-RC-4-Running-Shoes138960463_3.jpg') # ok
#predict('test_model/xxx.jpg') # ok

# Images of my dirty shoes sent via facebook messenger
#predict('test_model/adok2.jpg') # not ok
#predict('test_model/an2.jpg') # ok
#predict('test_model/ans.jpg') # ok
#predict('test_model/aok.jpg') # not ok
#predict('test_model/nbtr.jpg') # not ok
#predict('test_model/nbtr22.jpg') # not ok
#predict('test_model/nipg32.jpg') # not ok
# predict('test_model/nkkt7.jpg') # not ok
# predict('test_model/nzf3.jpg') # ok
# predict('test_model/zfl32.jpg') # not ok

# images from internet
# predict('test_model/nb_trail.jpeg') # not ok
# predict('test_model/pegasus-trail-2.jpg') # not ok
# predict('test_model/nike-air-zoom-rival-fly-3-471079-ct2405-358.jpeg') # not ok
