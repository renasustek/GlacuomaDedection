
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import PIL.Image
import tensorflow_datasets as tfds
import pathlib


plt.rcParams['figure.figsize'] = (7,7) 


dataset_url = 'http://vision.roboslang.org/open_datasets/Fundus_images.zip'




out_path = '/content/sample_data/'
archive = tf.keras.utils.get_file(origin=dataset_url, cache_dir='/content/sample_data/', extract=True)

data_dir = pathlib.Path(archive).with_suffix('')


print("Train:",len(list(data_dir.glob('./Train/Glaucoma_Positive/*.jpg'))) ,"|||", "Test:",len(list(data_dir.glob('./Test/Glaucoma_Positive/*.jpg'))))


positiveTrain = len(list(data_dir.glob('./Train/Glaucoma_Positive/*.jpg')))
negativeTrain = len(list(data_dir.glob('./Train/Glaucoma_Negative/*.jpg')))

postiveTest = len(list(data_dir.glob('./Test/Glaucoma_Positive/*.jpg')))
negativeTest= len(list(data_dir.glob('./Test/Glaucoma_Negative/*.jpg')))

print("Train positive:",positiveTrain ,"|||||||||", "Train negative:",negativeTrain)
print("Test positive:",postiveTest ,"|||||||||", "Test negative:",negativeTest)




positive_images = list(data_dir.glob('Train/Glaucoma_Positive/*'))

PIL.Image.open(str(positive_images[0]))

positive = PIL.Image.open(str(list(data_dir.glob('Test/Glaucoma_Positive/*.jpg'))[0]))
negative = PIL.Image.open(str(list(data_dir.glob('Test/Glaucoma_Negative/*.jpg'))[0]))

plt.figure(figsize=(15, 10))

plt.subplot(1, 2, 1)
plt.imshow(positive)

plt.subplot(1, 2, 2)
plt.imshow(negative)

plt.show()
e = 32

img_height = 256
img_width = 256



train_data_dir  = os.path.join(data_dir,'Train')
test_data_dir = os.path.join(data_dir,'Test')


seedVar = 123

train_ds = tf.keras.utils.image_dataset_from_directory(
    train_data_dir,
    validation_split=0.2,
    subset='training',
    seed=seedVar,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    train_data_dir,
    validation_split=0.2,
    subset='validation',
    seed=seedVar,
    image_size=(img_height, img_width),
    batch_size=batch_size
)



normalization_layer = tf.keras.layers.Rescaling(1./255)

normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]

print(np.min(first_image), np.max(first_image), first_image[0][0])

num_classes = 2


modelOne = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(6, (5, 5), activation='relu', input_shape=(img_height, img_width, 3)),
    tf.keras.layers.AveragePooling2D((2, 2)),
    tf.keras.layers.Conv2D(16, (5, 5), activation='relu'),
    tf.keras.layers.AveragePooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(120, activation='relu'),
    tf.keras.layers.Dense(84, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

modelTwo = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(96, (11, 11), strides=4, activation='relu', input_shape=(img_height, img_width, 3)),
    tf.keras.layers.MaxPooling2D((3, 3), strides=2),
    tf.keras.layers.Conv2D(256, (5, 5), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D((3, 3), strides=2),
    tf.keras.layers.Conv2D(384, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.Conv2D(384, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D((3, 3), strides=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(4096, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(4096, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

baseModel = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights='imagenet',
    input_shape=(img_height, img_width, 3)
)

modelThree = tf.keras.models.Sequential([
  baseModel,
  tf.keras.layers.GlobalAveragePooling2D(),
  tf.keras.layers.Dropout(0.5),
  tf.keras.layers.Dense(num_classes, activation='softmax')
])

modelOne.compile(
  optimizer='adam',
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
  metrics=['accuracy'])

modelTwo.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

modelThree.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

trainModelOne = modelOne.fit(train_ds, epochs=5, verbose=1, validation_data=val_ds)

trainModelTwo = modelTwo.fit(train_ds, epochs=5, verbose=1, validation_data=val_ds)

trainModelThree = modelThree.fit(train_ds, epochs=5, verbose=1, validation_data=val_ds)

train_loss1 = trainModelOne.history['loss']
val_loss1 = trainModelOne.history['val_loss']

train_loss2 = trainModelTwo.history['loss']
val_loss2 = trainModelTwo.history['val_loss']

train_loss3 = trainModelThree.history['loss']
val_loss3 = trainModelThree.history['val_loss']

epochs = range(1, len(train_loss1) + 1)

plt.figure(figsize=(10, 8))

plt.plot(epochs, train_loss1, label='1 - Training Loss', color='red', linestyle='-', marker='o')
plt.plot(epochs, val_loss1, label=' 1 - Validation Loss', color='red', linestyle='--', marker='x')

plt.plot(epochs, train_loss2, label='2 - Training Loss', color='green', linestyle='-', marker='o')
plt.plot(epochs, val_loss2, label='2 - Validation Loss', color='green', linestyle='--', marker='x')

plt.plot(epochs, train_loss3, label='3 - Training Loss', color='blue', linestyle='-', marker='o')
plt.plot(epochs, val_loss3, label='3 - Validation Loss', color='blue', linestyle='--', marker='x')

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Epochs for All Models')
plt.grid(True)
plt.legend()

plt.show()

test_ds = tf.keras.utils.image_dataset_from_directory(
  test_data_dir,
  image_size=(img_height, img_width),
  batch_size=batch_size)

test_loss1, test_accuracy1 = modelOne.evaluate(test_ds, verbose=1)
test_loss2, test_accuracy2 = modelTwo.evaluate(test_ds, verbose=1)
test_loss3, test_accuracy3 = modelThree.evaluate(test_ds, verbose=1)

