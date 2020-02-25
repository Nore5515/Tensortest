# Also, for future reference, in Atom use command+comment slash to comment lines.

from __future__ import absolute_import, division, print_function, unicode_literals


import tensorflow as tf

import matplotlib.pyplot as plt

import numpy as np

mnist = tf.keras.datasets.mnist

# x_train and y_train are a list of 60,000 uh...things.
# x_test and y_test are a list of 10,000...things.
# OH! I think they're images! I'll check using the image reader.
# confirm; they are hand drawn numbers, with y as labels
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



"""
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])
"""

# Creates a random NumPY Tensor with 1 image of 32x32 pixels with 3 colors
# If you try to replace the 3 with anything but 3, it'll error! no duh!
input_image = np.random.rand(1, 32, 32, 3)
print("RANDOM NUMPY TENSOR")
#print (input_image)



# huh?
# I believe this generates an empty "figure" state
plt.figure()
# shows an image from a numpy array
# Takes in some Numpy array of (H, W, 3)
# H is the height of the image in pixels, W is the width of the pixels and 3 is the RGB values
plt.imshow(input_image[0])
# presents a colorbar besides the image
plt.colorbar()
# disables the overlay grid
plt.grid(False)
# show the thing!
# plt.show()


#model = tf.keras.Sequential()

# Okay! This uh...creates tuple of integers.
# Specifically, this creates x as a placeholder tensor/array.
#x = tf.keras.Input(shape=(32,))
# Shape is simple! All it is, is the ranks of the arrays within it!
# Typically, for images, it's [Images, Colors(RGB), Height, Width]
# Traditionally, the first value is usually the batch, or the amount there is
# That's why you see "None" in front of all that stuff!
x = tf.keras.Input(shape=(32,32,3), batch_size=1)
data = [[1,2],[3,4],[5,6],[7,8],[9,10]]
print (data)

print ("--------------")
print (x.shape)
print ("--------------")
#tensor = ''.join(map(str, data))
#x.tensor = int(tensor)
#xLabels = tf.constant([1,2,3,4,5])

# This, I believe, creates a Dense layer using x as it's inputs!
# 16 is the "dimensionality of the output space"
# I'm assuming that just means it has 16 outputs?
# activation is the "activation" function to use.
# Otherwise it's just the linear activation function (whatever THAT means)
# Softmax, okay! It normalizes the values. NBD. Although what it has to do with activation, I still don't know.
# OOOOOH, maybe it uh, normalizes all the outputs!
# You can also specify the input array "shape" with input_shape=(16,)
y = tf.keras.layers.Dense(3, input_shape = (32,32,3))(x)
#y = tf.keras.Input([[0,1][0,1][0,1][0,1][0,1]])

# You can chain layers like this too!
z = tf.keras.layers.Dense(3, input_shape = (32,32,3)) (y)

# creates a model with X as it's input, and y as it's...labels?
# honestly not sure. Maybe y is it's output.
# Pretty sure X is it's input though.
# Okay just looked it up, Model seems to be a layer organizer.
# Makes sense, I guess. It also is more of a class that functions are run on.
# check the tensorflow docs about tf.keras.Model! Lots of deets.
model = tf.keras.Model(inputs=x, outputs=z)
#model = tf.keras.models.Sequential([x,y])
# HAAAALELUJA
print (model.summary())

# Alright, prepare for some REAL KNOWLEDGE about this next line
# But first, let's review the whole process of making this thing.
# 1) We take in our Inputs, and run them through a Linear Activation function.
# This means that we multiply the weights by the inputs, then add a bias.
# 2) We softmax the modified inputs, putting them all on a scale from 1 to 0
# This softmax value is roughly the same as the network's accuracy.
# 3) We then compare the softmax'd value vector/array/logits/whatever to the actual class of the inputs
# We do something called Cross Entropy, comparing the two and getting, roughly, how far off we were.
# 4) Tada! We now have accuracy and loss!
# Alright, that's fine and dandy, but what does this have to do with Compile?
# Wheelllllllp, I'll tell ya.
# Compile needs, at minimum, an Optimizer and a Loss...thing.
# Not sure what a "loss" is, but apparently it can be the name of an "objective function", an ACTUAL objective function or a Loss instance.
# The optimizer has some nice default, so we'll just be using that.
# Oooooor not! So we'll need to make an optimizer! Adam seems nice.
# Which means we need to get a loss function! Or loss objective function. Whatever.
# BinaryCrossentropy sounds...good? I think?

loss = tf.keras.losses.BinaryCrossentropy()
#loss = tf.losses.Loss()
#optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
model.compile(optimizer, loss)

# holy crap I can't believe compile actually worked.
# Well, onto to the next stage, I guess
# Annnd Fit doesn't work. go figure.

testArray = np.random.rand(1, 32, 32, 3)
#testArray = np.array([100])

print ("==============\nINPUT SHAPE: ")
print (input_image.shape)
print ("X INPUT SHAPE: ")
print (x.shape)
print ("TEST ARRAY SHAPE: ")
print (testArray.shape)
#print ("")

# runs it thorugh input_image with the labels testArray
# does so 200 times, trying to teach it
model.fit(input_image, testArray, epochs=200)

print ("NOW TESTING IT...")

# tests it!
#model.evaluate(input_image, testArray)

# I stole this from online bc i don't wanna import anything else
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()



# Now, compare what it THINKs the output should be, to the actual outputs
# It will display the actual output, then the predicted output.
print ("\nUNEDITED PREDICTION\n")
print (model.predict(input_image)[0])
print ("\nSOFTMAXXED PREDICTION\n")
print (softmax(model.predict(input_image)[0]))



plt.imshow(testArray[0])
#plt.show()

plt.imshow(
    softmax(model.predict(input_image)[0])
)
plt.show()



#print (model.summary())


print ("Hello World!")
