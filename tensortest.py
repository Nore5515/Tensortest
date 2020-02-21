# Also, for future reference, in Atom use command+comment slash to comment lines.

from __future__ import absolute_import, division, print_function, unicode_literals


import tensorflow as tf

import matplotlib.pyplot as plt


mnist = tf.keras.datasets.mnist

# x_train and y_train are a list of 60,000 uh...things.
# x_test and y_test are a list of 10,000...things.
# OH! I think they're images! I'll check using the image reader.
# confirm; they are hand drawn numbers.
(x_train, y_train), (x_test, y_test) = mnist.load_data()


# print(len(x_train))
# print(len(y_train))
# print(len(x_test))
# print(len(y_test))

# dividing them doesn't seem to change much.
# I'm not sure why we're doing this.
# and why only the x ones???

# I see! The Y tuples/arrays is a list of numbers!
# It's what each image corresponds to!
#print (y_train)

# OOOOOOH WE'RE DIVIDING EVERY VALUE IN THE ARRAYS/TUPLES BY 255
# THAT WAY THEIR RBG VALUE OF EACH TILE IS ONLY BETWEEN 1 AND 0

# print (x_train[0])

x_train, x_test = x_train / 255.0, x_test / 255.0

# print (x_train[0])

# print(len(x_train))
# print(len(y_train))
# print(len(x_test))
# print(len(y_test))

# huh?
# I believe this generates an empty "figure" state
plt.figure()
# shows an image? but not really? from a numpy array?
# Takes in some Numpy array of (M, N, 3)
# M is all the pixels, N is each indivdual pixel and 3 is the RGB values
plt.imshow(x_train[0])
# presents a colorbar besides the image
plt.colorbar()
# disables the overlay grid
plt.grid(False)
# show the thing!
#plt.show()

"""
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])
"""

#model = tf.keras.Sequential()

# Okay! This uh...creates tuple of integers.
# Specifically, that it will be an inpu t
x = tf.keras.Input(shape=(32,))
y = tf.keras.layers.Dense(16, activation='softmax')(x)


print ("Hello World!")
