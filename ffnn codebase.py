import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_rem, y_train, y_rem = train_test_split(x_train,y_train, train_size=0.9)
d1,d2,d3,d4=x_train.shape
x_train=x_train.reshape(d1,-1)
d1,d2,d3,d4=x_test.shape
x_test=x_test.reshape(d1,-1)
d1,d2,d3,d4=x_rem.shape
x_rem=x_rem.reshape(d1,-1)

i_size=x_train[0].shape
x = tf.keras.layers.Input(shape=(i_size[0],))
y = tf.keras.layers.Dense(1024)(x)
y = tf.keras.layers.Dense(512)(y)
y = tf.keras.layers.Dense(256)(y)
y = tf.keras.layers.Dense(128)(y)
y = tf.keras.layers.Dense(64)(y)
y = tf.keras.layers.Dense(32)(y)
y = tf.keras.layers.Dense(10,activation='softmax')(y)
model = tf.keras.models.Model(x, y)
model.compile(
   loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy']
)
model.fit(x_train, y_train, validation_data = (x_rem, y_rem),batch_size = 64, epochs = 50,verbose=2)
plt.plot(model.history.history['loss'])
plt.plot(model.history.history['val_loss'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
score = model.evaluate(x_test, y_test, verbose = 0) 
print('Test loss:', score[0]) 
print('Test accuracy:', score[1])

# print(model.summary())