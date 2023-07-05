import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import constants as constant
import preprocessing_data as pp
import drawCurves as drawer
import plotly.graph_objects as go
from plotly.offline import plot
import plotly.graph_objects as go


def plot_confusion_matrix(confusion_mtx, classes):

    fig = go.Figure(data=go.Heatmap(z=confusion_mtx,
                                    x=classes,
                                    y=classes,
                                    colorscale='Blues'))
    fig.update_layout(title='Confusion Matrix',
                      xaxis_title='Predicted label',
                      yaxis_title='True label',
                      xaxis=dict(type='category', automargin=True),
                      yaxis=dict(type='category', automargin=True),
                      autosize=True,
                      width=2000,
                      height=2000
                      )
    plot(fig, filename="confusion_matrix.html", auto_open=False)


def trainTheModelwithEarlyStop(model):

    # early_stopping = tf.keras.callbacks. EarlyStopping(monitor='val_accuracy', mode='max',
    #                                                    patience=10,  restore_best_weights=True)  # 10

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)


    history = model.fit(
        pp.train_generator,
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

    plot_confusion_matrix(confusion_mtx, traingen)


def mainModelCreationFunc():

    # main CNN layers
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
    model.add(tf.keras.layers.Dense(pp.train_generator.num_classes, activation='softmax'))

    # optimizer = tf.keras.optimizers.Adam(lr=0.0001)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # showing a summary for what is happening and change in images dimensions while every CNN layer.
    model.summary()

    # Early stopping function to avoid the overfitting effect

    # Model history is returned back
    his = trainTheModelwithEarlyStop(model)

    # Generate predictions on the test set and generating confusion matrix
    generatePredictionandPrintConfusionMatrix(
        model, list(pp.train_generator.class_indices.keys()))

    drawer.showLossCurves(history=his)
    drawer.showAccuracyCurves(history=his)

    # Save the model
    model.save('image_classification_model.h5')


mainModelCreationFunc()


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
