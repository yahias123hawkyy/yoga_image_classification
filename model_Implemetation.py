import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import constants as constant
import preprocessing_data as pp
import draw as drawer




def trainTheModelwithEarlyStop(model):

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='max', patience=5,  restore_best_weights=True)  # 10

    # early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)


    history = model.fit(

        ## pp is an instance from the implemented module preprocessing_data.py
        pp.train_generator,

        ## constant is an instance from the implemented module constants.py
        steps_per_epoch=pp.train_generator.n // constant.batch_size,
        epochs=constant.epochs,
        validation_data=pp.test_generator,
        validation_steps=pp.test_generator.n // constant.batch_size,
        callbacks=[early_stopping]
    )

    return history


def generatePredictionandPrintConfusionMatrix(model, traingen):
    pp.test_generator.reset()
    y_pred = model.predict(pp.test_generator)
    y_pred = np.argmax(y_pred, axis=1)
    true_labels = pp.test_generator.classes

    confusion_mtx = confusion_matrix(true_labels, y_pred)

    drawer.plot_confusion_matrix(confusion_mtx, traingen)







def mainModelCreationFunc():



    # To make the results the same, it controls the randoms genertaed by Numpy and Tensorflow
    np.random.seed(42)   
    tf.random.set_seed(42)

    # main CNN layers STack
    model = tf.keras.models.Sequential()


    model.add( tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(constant.image_size[0], constant.image_size[1], 3)))
    model.add( tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))



    # Flattening layer
    model.add(tf.keras.layers.Flatten())

    # Dense Layer
    model.add(tf.keras.layers.Dense(128, activation='relu'))


    # Dense Layer to make the Distributed Propability Function
    model.add(tf.keras.layers.Dense(pp.train_generator.num_classes, activation='softmax'))

    # optimizer = tf.keras.optimizers.Adam(lr=0.0001)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # showing a summary for what is happening and change in images dimensions while every CNN layer.
    model.summary()


    # Model history is returned back
    his = trainTheModelwithEarlyStop(model)

    # Generate predictions on the test set and generating confusion matrix
    generatePredictionandPrintConfusionMatrix(
        model, list(pp.train_generator.class_indices.keys()))


    ## ploting and drawing the training accuracy/loss Curves using the implemented module Drawer
    drawer.showLossCurves(history=his)
    drawer.showAccuracyCurves(history=his)



mainModelCreationFunc()



















#### IT WAS A TRIAL TO USE A TRANSFER LEARNING INSTEAD OF CREATING THE WHOLE CNN STACKS, TO INCREASE ACCURACY BUT IT WAS IN VAIN!
# Model compiling

#     base_model = tf.keras.applications.MobileNetV3Large(weights='imagenet', include_top=False, input_shape=(constant.image_size[0], constant.image_size[1], 3))

#     # Freeze the layers in the base model
#     base_model.trainable = True
# # Fine-tune from this layer onwards
#     fine_tune_at = 100
#     for layer in base_model.layers[:fine_tune_at]:
#         layer.trainable = False

#     # Create a new model on top of the pre-trained base model
#     model = tf.keras.models.Sequential([
#         base_model,
#         tf.keras.layers.GlobalAveragePooling2D(),
#         tf.keras.layers.Dense(128, activation='relu'),
#         tf.keras.layers.Dense(pp.train_generator.num_classes, activation='softmax')
#     ])
