import tensorflow as tf
import constants as constant



train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)


test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        constant.train_path,
        target_size=constant.image_size,
        batch_size=constant.batch_size,
        class_mode='categorical'
    )

test_generator = test_datagen.flow_from_directory(
       constant.test_path,
       target_size=constant.image_size,
        batch_size=constant.batch_size,
       class_mode='categorical',
       shuffle=False
    )