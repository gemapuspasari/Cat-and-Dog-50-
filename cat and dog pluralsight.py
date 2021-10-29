#!/usr/bin/env python
# coding: utf-8

# In[1]:


#https://www.pluralsight.com/guides/build-your-first-image-classification-experiment


# In[2]:


# Exploring the data
DATASET_LOCATION = "C:/Users/Gema Puspa Sari/Downloads/python_______latihan/cats-and-dogs/train"


# In[3]:


# Collect the labels and filenames of the datasets

import os

filenames = os.listdir(DATASET_LOCATION)
classes = []
for filename in filenames:
    image_class = filename.split(".")[0]
    if image_class == "dog":
        classes.append(1)
    else:
        classes.append(0)


# In[4]:


# Read the dataset into a pandas dataframe for convenient access
import pandas as pd

df = pd.DataFrame({"filename": filenames, "category": classes})
df["category"] = df["category"].replace({0: "cat", 1: "dog"})


# In[5]:


df.head()


# In[6]:


df.category.value_counts()


# In[7]:


import random
from keras.preprocessing.image import load_img
import matplotlib.pyplot as plt

sample1 = random.choice(filenames)
image1 = load_img(DATASET_LOCATION + "/" + sample1)
plt.imshow(image1)


# In[8]:


sample2 = random.choice(filenames)
image2 = load_img(DATASET_LOCATION + "/" + sample2)
plt.imshow(image2)


# In[9]:


IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
INPUT_SHAPE = (IMAGE_WIDTH, IMAGE_HEIGHT, 1)


# In[10]:


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=INPUT_SHAPE))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(2, activation="softmax"))

model.compile(
    loss=keras.losses.categorical_crossentropy,
    optimizer=keras.optimizers.Adadelta(),
    metrics=["accuracy"],
)


# In[11]:


from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(df, test_size=0.20, random_state=42)


# In[12]:


from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rotation_range=15,
    rescale=1.0 / 255,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
)


# In[13]:


BATCH_SIZE = 16
train_generator = train_datagen.flow_from_dataframe(
    train_df,
    DATASET_LOCATION,
    x_col="filename",
    y_col="category",
    target_size=IMAGE_SIZE,
    class_mode="categorical",
    batch_size=BATCH_SIZE,
    color_mode="grayscale",
)


# In[14]:


DATATEST_LOCATION = "C:/Users/Gema Puspa Sari/Downloads/python_______latihan/cats-and-dogs/test1"


# In[15]:


test_datagen = ImageDataGenerator(
    rotation_range=15,
    rescale=1.0 / 255,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
)


# In[16]:


BATCH_SIZE = 16
test_generator = test_datagen.flow_from_dataframe(
    test_df,
    DATATEST_LOCATION,
    x_col="filename",
    y_col="category",
    target_size=IMAGE_SIZE,
    class_mode="categorical",
    batch_size=BATCH_SIZE,
    color_mode="grayscale",
)


# In[17]:


example_df = train_df.sample(n=1)
example_generator = train_datagen.flow_from_dataframe(
    example_df,
    DATASET_LOCATION,
    x_col="filename",
    y_col="category",
    target_size=IMAGE_SIZE,
    class_mode="categorical",
    color_mode="grayscale",
)


# In[18]:


plt.figure(figsize=(12, 12))
for i in range(0, 15):
    plt.subplot(5, 3, i + 1)
    for X_batch, Y_batch in example_generator:
        image = X_batch[0]
        image = image.reshape(IMAGE_SIZE)
        plt.imshow(image)
        break
plt.tight_layout()
plt.show()


# In[19]:


EPOCHS = 10
history = model.fit_generator(
    train_generator,
    epochs=EPOCHS,
    validation_data=test_generator,
    validation_steps=test_df.shape[0] // BATCH_SIZE,
    steps_per_epoch=train_df.shape[0] // BATCH_SIZE,
)


# In[24]:


NUM_SAMPLES = 10
sample_test_df = test_df.head(NUM_SAMPLES).reset_index(drop=True)
sample_test_datagen = ImageDataGenerator(rescale=1.0 / 255)
sample_test_generator = sample_test_datagen.flow_from_dataframe(
    sample_test_df,
    DATASET_LOCATION,
    x_col="filename",
    y_col="category",
    target_size=IMAGE_SIZE,
    class_mode="categorical",
    batch_size=BATCH_SIZE,
    color_mode="grayscale",
)


# In[25]:


predict = model.predict_generator(sample_test_generator)
import numpy as np

predictions = np.argmax(predict, axis=-1)


# In[26]:


plt.figure(figsize=(12, 24))
for index, row in sample_test_df.iterrows():
    filename = row["filename"]
    prediction = predictions[index]
    img = load_img(DATASET_LOCATION + "/" + filename)
    plt.subplot(6, 3, index + 1)
    plt.imshow(img)
    plt.xlabel(prediction)
plt.tight_layout()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




