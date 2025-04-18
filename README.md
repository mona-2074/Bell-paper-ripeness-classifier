# Bell-paper-ripeness-classifier
from tensorflow import keras, cast, float32
from keras import Sequential, layers
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
dataset = keras.preprocessing.image_dataset_from_directory(
    directory = "/content/drive/MyDrive/Bell_Paper_Images/",\
    image_size = (256, 256),
    batch_size = 32,
    shuffle = True
    )
    class_list = dataset.class_names
class_list[0] = "Infected Leaf"
class_list[1] = 'Healthy Leaf'
class_list
for image_batch, label_batch in dataset.take(1):
  plt.imshow(image_batch[0].numpy().astype('uint8'))
  plt.axis('off')
  def process(image, label):
  image = cast(image/255., float32)
  return image, label

dataset = dataset.map(process)
train_ds = dataset.take(50)
test_ds = dataset.skip(50)
valid_ds = test_ds.take(6)
test_ds = test_ds.skip(6)
augmentation = Sequential([
    layers.experimental.preprocessing.RandomFlip('horizontal_and_vertical'),
    layers.experimental.preprocessing.RandomRotation(0.2),
    layers.experimental.preprocessing.RandomZoom(0.2)
])
model = Sequential(augmentation)

model.add(Conv2D(32, (3, 3), activation = 'relu', input_shape = (32, 256, 256, 3)))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(64, (3, 3), activation = 'relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(64, (3, 3), activation = 'relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(64, (3, 3), activation = 'relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(64, (3, 3), activation = 'relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())

model.add(Dense(64, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

model.build((32, 256, 256, 3))
model.summary()
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
history = model.fit(train_ds, epochs = 12, batch_size = 32, validation_data = valid_ds)
for image_batch, label_batch in test_ds.take(1):
  plt.figure(figsize = (20, 12))
  for i in range(8):
      ax = plt.subplot(2, 4, i+1)
      test_image = image_batch[i].numpy()
      test_label = label_batch[i].numpy()
      test_image = np.expand_dims(test_image, axis=0)
      x = round(model.predict(test_image)[0][0])
      plt.title("Actual Label: " + class_list[test_label] + "\n" + "Predicted Label: " + class_list[x])
      plt.imshow(image_batch[i].numpy())
      plt.axis('off')
      path = input("Enter path: ")
try:
  img = image.load_img(path, target_size = (256, 256))
  test_image = image.img_to_array(img)
except FileNotFoundError:
  print("A file with this name/path doesn't exist. Please enter an existing file")
except:
  print("The file isn't in the desired format. Please make sure it's in png, jpg or jpeg")


test_image = test_image/255
test_image = np.expand_dims(test_image, axis = 0)
x = round(model.predict(test_image)[0][0])
plt.imshow(img)
plt.axis('off')
plt.title("Predicted Label: " + class_list[x])
