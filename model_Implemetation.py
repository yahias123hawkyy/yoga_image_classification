import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import constants as constant
import preprocessing_data as pp
import drawCurves as drawer 

def trainTheModelwithEarlyStop(model):


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



def generatePredictionandPrintConfusionMatrix(model):
    pp.test_generator.reset()
    y_pred = model.predict(pp.test_generator)
    y_pred = np.argmax(y_pred, axis=1)
    true_labels = pp.test_generator.classes

    confusion_mtx = confusion_matrix(true_labels, y_pred)
    
    print("Confusion Matrix:")
    print(confusion_mtx)









def mainModelCreationFunc():

    # main CNN layers
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(constant.image_size[0], constant.image_size[1], 3)))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    
    #Flattening layer
    model.add(tf.keras.layers.Flatten())
    
    #Dense Layer
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(pp.train_generator.num_classes, activation='softmax'))


    #Model compiling
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    
    ## showing a summary for what is happening and change in images dimensions while every CNN layer.
    model.summary()




    ## Early stopping function to avoid the overfitting effect
    
    # Model history is returned back 
    his = trainTheModelwithEarlyStop(model)


    

   
   
    
    # Generate predictions on the test set and generating confusion matrix
    generatePredictionandPrintConfusionMatrix(model)
    
    
    drawer.showLossCurves(history=his)
    drawer.showAccuracyCurves(history=his)

    

    # Save the model
    model.save('image_classification_model.h5')


mainModelCreationFunc()




   