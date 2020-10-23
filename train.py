import os
import numpy as np
from keras import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

TEST=False

SEGMENTED_DIR = os.path.abspath("img/segmented")
image_generator = ImageDataGenerator(
    validation_split=0.2,
    rescale=1.0 / 255,
    dtype=np.float32
)

training_data = image_generator.flow_from_directory(
    SEGMENTED_DIR,
    target_size=(400, 400),
    color_mode="grayscale",
    seed=1237,
    subset="training",
    batch_size=32
    #save_to_dir=os.path.abspath("img/segmented_preprocessed")
)

model = Sequential()
# first layer output is 100, kernel size is 3x3, input image shape is 400 w, 400 h, 1 channel
model.add(Conv2D(96, kernel_size=(3, 3), input_shape=(400, 400, 1), activation="relu"))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(96, kernel_size=(3, 3), activation="relu"))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(192, kernel_size=(3, 3), activation="relu"))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(192, kernel_size=(3, 3), activation="relu"))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())
model.add(Dense(216, activation="relu"))
#model.add(Dense(125, activation="relu"))
model.add(Dense(356, activation="relu"))

model.add(Dense(6, activation="softmax"))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(training_data, epochs=50, batch_size=32, verbose=1)


"""
X_validation, y_validation, *other = image_generator.flow_from_directory(
    SEGMENTED_DIR,
    target_size=(400, 400),
    color_mode="grayscale",
    seed=1237,
    subset="validation",
    #save_to_dir=os.path.abspath("img/segmented_preprocessed")
)
"""
