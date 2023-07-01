import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import test_model as test


train_path = 'D:/Image_classification_yoga_poses/train_dataset'
test_path = 'D:/Image_classification_yoga_poses/test_dataset'

image_size = (128, 128)
batch_size = 32


def modelCreation():

    # Define image size and batch size

    # Create data generators with data augmentation for training and testing
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    test_generator = test_datagen.flow_from_directory(
        test_path,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    # Create a CNN model
    # Create a CNN model
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_size[0], image_size[1], 3)))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(train_generator.num_classes, activation='softmax'))


    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    epochs = 50
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.n // batch_size,
        epochs=epochs,
        validation_data=test_generator,
        validation_steps=test_generator.n // batch_size
    )

    # Generate predictions on the test set
    test_generator.reset()

    y_pred = model.predict(test_generator)
    y_pred = np.argmax(y_pred, axis=1)

    # Obtain true labels for the test set
    true_labels = test_generator.classes

    #  confusion matrix
    confusion_mtx = confusion_matrix(true_labels, y_pred)

    # Plot the training curves
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Training Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(['Training Accuracy', 'Validation Accuracy'])
    plt.show()

    # Print the confusion matrix
    print("Confusion Matrix:")
    print(confusion_mtx)

    # Save the model
    model.save('image_classification_model.h5')


test_image_path_model = 'D:/Image_classification_yoga_poses/testing_images/29-0.png'


# modelCreation()

test.testThemodelByAnyImage(train_path=train_path,
                            test_image_path=test_image_path_model)
