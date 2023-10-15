import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px 
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications import DenseNet121
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, MaxPooling2D 
from tensorflow.keras.models import Sequential

import warnings
warnings.filterwarnings('ignore')



train_dir = "C:/Users/niam2/OneDrive/Desktop/project/Dataset/level/training" 
test_dir = "C:/Users/niam2/OneDrive/Desktop/project/Dataset/level/validation"

SEED = 12
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 128
EPOCHS = 5
IMG_SHAPE = (IMG_HEIGHT, IMG_WIDTH)
INPUT_SHAPE=(IMG_WIDTH,IMG_HEIGHT,3)
LR =0.00003

crime_types=os.listdir(train_dir)
n=len(crime_types)
print("Number of crime categories : ",n)


train_set=image_dataset_from_directory(
train_dir,
label_mode="categorical",
batch_size=BATCH_SIZE,
image_size=IMG_SHAPE,
shuffle=True,
seed=SEED,
validation_split=0.2,
subset="training",
)


val_set=image_dataset_from_directory(
train_dir,
label_mode="categorical",
batch_size=BATCH_SIZE,
image_size=IMG_SHAPE,
shuffle=True,
seed=SEED,
validation_split=0.2,
subset="validation",
)



test_set=image_dataset_from_directory(
test_dir,
label_mode="categorical",
class_names=None,
batch_size=BATCH_SIZE,
image_size=IMG_SHAPE,
shuffle=False,
seed=SEED,
)


def transfer_learning():
    base_model=DenseNet121(include_top=False,input_shape=INPUT_SHAPE,weights="imagenet")
    thr=149
    for layers in base_model.layers[:thr]: layers.trainable=False
    for layers in base_model.layers[thr:]: layers.trainable=False
    return base_model



def create_model(): 
    model=Sequential()
    base_model=transfer_learning()
    model.add(base_model)
    model.add(GlobalAveragePooling2D())
    model.add(Dense (256, activation="relu")) 
    model.add(Dropout (0.2))
    model.add(Dense (512, activation="relu")) 
    model.add(Dropout (0.2))
    model.add(Dense (1024, activation="relu"))
    model.add(Dense (n, activation="softmax"))
    model.summary()
    return model


model=create_model()
model.compile(optimizer="adam",
loss='categorical_crossentropy', metrics = ['accuracy'])


# Train the model
history = model.fit(x=train_set, epochs=EPOCHS, validation_data=val_set)

#save
model.save("level.h5")

#Testing the model
model.load_weights('level.h5')
y_true=np.array([])

for x,y in test_set:
    y_true=np.concatenate([y_true,np.argmax(y.numpy(),axis=-1)])

y_predict=model.predict(test_set)