from PIL import Image
import numpy as np
import tensorflow as tf




def testThemodelByAnyImage(train_path,test_image_path):


            model = tf.keras.models.load_model('image_classification_model.h5')


            image_size = (128, 128)
            batch_size = 32

            train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                rescale=1./255,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True
            )

            train_generator = train_datagen.flow_from_directory(
                train_path,
                target_size=image_size,
                batch_size=batch_size,
                class_mode='categorical'
            )


            

            # Load and preprocess the test image
            test_image = tf.keras.preprocessing.image.load_img(test_image_path, target_size=image_size)
            test_image = tf.keras.preprocessing.image.img_to_array(test_image)
            test_image = np.expand_dims(test_image, axis=0)
            test_image = test_image / 255.0

            # Make predictions on the test image
            prediction = model.predict(test_image)
            predicted_class = np.argmax(prediction)

            # Get the class labels from the generator
            class_labels = list(train_generator.class_indices.keys())

            # Get the predicted yoga pose name
            predicted_pose = class_labels[predicted_class]

            # Print the predicted yoga pose
            print("Predicted Yoga Pose:", predicted_pose)

