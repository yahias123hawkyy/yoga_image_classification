from PIL import Image
import numpy as np
import tensorflow as tf
import constants as constant
import preprocessing_data as pp




def testThemodelByAnyImage(test_image_path):


            model = tf.keras.models.load_model('image_classification_model.h5')

            

            # Load and preprocess the test image
            test_image = tf.keras.preprocessing.image.load_img(test_image_path, target_size=constant.image_size)
            test_image = tf.keras.preprocessing.image.img_to_array(test_image)
            test_image = np.expand_dims(test_image, axis=0)
            test_image = test_image / 255.0

            # Make predictions on the test image
            prediction = model.predict(test_image)
            predicted_class = np.argmax(prediction)

            # Get the class labels from the generator
            class_labels = list(pp.train_generator.class_indices.keys())

            # Get the predicted yoga pose name
            predicted_pose = class_labels[predicted_class]

            # Print the predicted yoga pose
            print("Predicted Yoga Pose is ", predicted_pose)

