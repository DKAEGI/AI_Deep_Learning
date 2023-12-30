from keras.datasets import mnist # Import dataset with handwritten letters
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical # is used to convert integer labels (which are often used to represent different classes or categories in machine learning tasks) into a binary matrix representation. This process is known as one-hot encoding.
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from sklearn.metrics import classification_report

# load the data
# Returns 4 arrays: x_train, y_train, x_test, and y_test. Dividend into 2 tuples (1 for train and 1 for test).
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Visualize data
single_image = x_train[0]
plt.imshow(single_image) # inside jupyter lab
plt.show() # outside jupyter lab

# Labels y_train = array([5, 0, 4, ..., 5, 6, 8], dtype=uint8) and y_test = array([7, 2, 1, ..., 4, 5, 6], dtype=uint8)
# labels are literally categories of numbers. We need to translate this to be "one hot encoded" so our CNN can understand, otherwise it will think this is some sort of regression problem on a continuous axis.
# One-Hot Encoding: In one-hot encoding, integer labels are converted into a binary matrix in such a way that:
# Each column in the matrix corresponds to a class label.
# For each instance (row in the matrix), only the column corresponding to its label is set to 1, and all other columns are set to 0.
y_cat_test = to_categorical(y_test,10) # y_test array of integer labels
y_cat_train = to_categorical(y_train,10) # 10 is the number of classes in the one-hot encoded matrix,ten digits (0 through 9). This number should match the number of distinct categories in the dataset.

# Processing X Data
# Normalize the X data
# single_image.max() = 255, single_image.min() = 0
x_train = x_train/255 # divided by x_train_max
x_test = x_test/255
scaled_single = x_train[0]
# scaled_single.max() = 1
plt.imshow(scaled_single) # inside jupyter lab
plt.show() # outside jupyter lab

# Reshaping the Data
# Right now our data is 60,000 images stored in 28 by 28 pixel array formation.
# This is correct for a CNN, but we need to add one more dimension to show we're dealing with 1 RGB channel (since technically the images are in black and white, only showing values from 0-255 on a single channel), an color image would have 3 dimensions.
x_train = x_train.reshape(60000, 28, 28, 1) # x_train.shape before (60000, 28, 28), and after (60000, 28, 28, 1)
x_test = x_test.reshape(10000,28,28,1)


###########################################################################################################################
# TRAINING THE MODEL


### Create Model ###
model = Sequential() # Linear stack of layers, where each layer has exactly one input tensor and one output tensor.

# CONVOLUTIONAL LAYER
model.add(Conv2D(filters=32, kernel_size=(4,4),input_shape=(28, 28, 1), activation='relu',))
# 2D convolutional layer to the model
# filters=32:  filter (or kernel) matrix used to extract features from input images through a process called convolution. Number 32 is often used as a starting point in many CNN architectures, particularly for small to medium-sized datasets.
# kernel_size=(4,4): Specifies the height and width of the 2D convolution window.
# input_shape=(28, 28, 1): The shape of the input data: height, width, and channels of the images. In this case, it's set for 28x28 pixel images with 1 color channel (grayscale).
# activation='relu': The activation function used. ReLU (Rectified Linear Unit) is a common activation function in neural networks.

# POOLING LAYER
model.add(MaxPool2D(pool_size=(2, 2))) # Reduces the spatial dimensions

# FLATTEN IMAGES FROM 28 by 28 to 764 BEFORE FINAL LAYER
model.add(Flatten()) # This layer flattens the multi-dimensional output of the previous layers into a one-dimensional array. This is necessary because fully connected layers (like the Dense layer that follows) expect 1D arrays as input.

# 128 NEURONS IN DENSE HIDDEN LAYER (YOU CAN CHANGE THIS NUMBER OF NEURONS)
model.add(Dense(128, activation='relu')) # This adds a fully connected layer with 128 neurons to the network. The relu activation function is used here as well.

# LAST LAYER IS THE CLASSIFIER, THUS 10 POSSIBLE CLASSES
model.add(Dense(10, activation='softmax')) # The softmax activation function is used to turn the output into probability scores for each class.

# The loss function used is categorical crossentropy, common for multi-class classification tasks.
# The optimizer used is RMSprop, which is an adaptive learning rate optimizer.
# The model will track accuracy during training.
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

print(model.summary())


### Train Model ###
model.fit(x_train,y_cat_train,epochs=10) # Can take a while, depending on cpu. 


### Evaluate Model ###
print(model.metrics_names)
model.evaluate(x_test,y_cat_test)
predictions = model.predict_classes(x_test)
print(classification_report(y_test,predictions))
# Precision: This is the ratio of correctly predicted positive observations to the total predicted positives. It answers the question: "Of all the instances the model labeled as positive, how many are actually positive?"
# Recall (Sensitivity): This is the ratio of correctly predicted positive observations to all the observations in the actual class. It answers: "Of all the actual positives, how many did the model correctly identify?"
# F1-Score: This is the weighted average of Precision and Recall. It takes both false positives and false negatives into account, and is a good way to summarize the model’s performance in a single metric, especially when dealing with imbalanced datasets.
# Support: This refers to the number of actual occurrences of the class in the specified dataset. It's useful for seeing the distribution of classes in the dataset.


### Save Model ###
model.save('mnist_test.h5')
