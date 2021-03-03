import tensorflow as tf
import datetime
import os.path

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

"""
run (UNIX): python -m MNIST.train
run (WINDOWS as Admin in command line): py -3.7 -m MNIST.train

"""

# get the data
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images.astype('float32') / 255.0, test_images.astype('float32') / 255.0
train_images = tf.expand_dims(train_images, axis=3)
test_images = tf.expand_dims(test_images, axis=3)

# create the model
inputs = tf.keras.Input(shape=(28, 28, 1), name='digits')
x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Conv2D(64, (3, 3), activation='relu')(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Conv2D(64, (3, 3), activation='relu')(x)
x = layers.Flatten()(x)
x = layers.Dense(64, activation='relu')(x)
x = layers.Dense(10, name='dense_logits')(x)
outputs = layers.Activation('softmax', dtype='float32', name='predictions')(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

# Compile and train the model
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=tf.keras.optimizers.RMSprop(),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels,
                    batch_size=256,
                    epochs=50,
                    validation_data=(test_images, test_labels))

# evaluate
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print(test_acc)