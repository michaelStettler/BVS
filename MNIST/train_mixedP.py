import tensorflow as tf
import datetime
import os.path

from tensorflow.keras import datasets, layers, models
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import matplotlib.pyplot as plt

"""
run (UNIX): python -m MNIST.train
tensorboard: tensorboard --logdir=log/fit_MNIST --host localhost --port 8088
run (WINDOWS as Admin in command line): py -3.7 -m MNIST.train
tensorboard --logdir=C:\ Users\Michael\PycharmProjects\BVS\logs\ fit_MNIST --host localhost --port 8088
then in browser go to: http://localhost:8088/
"""

# set the policy
policy = mixed_precision.Policy('mixed_float16')
# policy = mixed_precision.Policy('float32')
mixed_precision.set_policy(policy)
print('Compute dtype: %s' % policy.compute_dtype)
print('Variable dtype: %s' % policy.variable_dtype)

mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize pixel values to be between 0 and 1
# train_images, test_images = train_images.astype('float32') / 255.0, test_images.astype('float32') / 255.0
# train_images = tf.expand_dims(train_images, axis=3)
# test_images = tf.expand_dims(test_images, axis=3)
train_images = train_images.reshape(60000, 784).astype('float32') / 255
test_images = test_images.reshape(10000, 784).astype('float32') / 255

# create the model
# inputs = tf.keras.Input(shape=(28, 28, 1), name='digits')
# x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
# x = layers.MaxPooling2D((2, 2))(x)
# x = layers.Conv2D(64, (3, 3), activation='relu')(x)
# x = layers.MaxPooling2D((2, 2))(x)
# x = layers.Conv2D(64, (3, 3), activation='relu')(x)
# x = layers.Flatten()(x)
# x = layers.Dense(64, activation='relu')(x)
# x = layers.Dense(10, name='dense_logits')(x)
# outputs = layers.Activation('softmax', dtype='float32', name='predictions')(x)
# model = tf.keras.Model(inputs=inputs, outputs=outputs)

inputs = tf.keras.Input(shape=(784,), name='digits')
if tf.config.list_physical_devices('GPU'):
  print('The model will run with 4096 units on a GPU')
  num_units = 4096
else:
  # Use fewer units on CPUs so the model finishes in a reasonable amount of time
  print('The model will run with 64 units on a CPU')
  num_units = 64
dense1 = layers.Dense(num_units, activation='relu', name='dense_1')
x = dense1(inputs)
dense2 = layers.Dense(num_units, activation='relu', name='dense_2')
x = dense2(x)
print('x.dtype: %s' % x.dtype.name)
# 'kernel' is dense1's variable
print('dense1.kernel.dtype: %s' % dense1.kernel.dtype.name)
x = layers.Dense(10, name='dense_logits')(x)
outputs = layers.Activation('softmax', dtype='float32', name='predictions')(x)
print('Outputs dtype: %s' % outputs.dtype.name)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

# Compile and train the model
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=tf.keras.optimizers.RMSprop(),
              metrics=['accuracy'])

log_dir = os.path.join("logs", "fit_MNIST", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                      histogram_freq=1,
                                                      # histogram_freq=1)
                                                      profile_batch='400, 600')

initial_weights = model.get_weights()
history = model.fit(train_images, train_labels,
                    batch_size=64,
                    epochs=5,
                    validation_data=(test_images, test_labels),
                    callbacks=[tensorboard_callback])

# evaluate
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print(test_acc)