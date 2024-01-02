from keras.datasets import cifar10 # Import dataset with 10 classes of colored images
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical # is used to convert integer labels (which are often used to represent different classes or categories in machine learning tasks) into a binary matrix representation. This process is known as one-hot encoding.
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from sklearn.metrics import classification_report

# load the data
# Returns 4 arrays: x_train, y_train, x_test, and y_test. Dividend into 2 tuples (1 for train and 1 for test).
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Visualize data
single_image = x_train[0]
plt.imshow(single_image) # inside jupyter lab
plt.show() # outside jupyter lab

# The images are labelled with one of 10 mutually exclusive classes: airplane, automobile (but not truck or pickup truck), bird, cat, deer, dog, frog, horse, ship, and truck (but not pickup truck). 
# Labels y_train[0] = array([6], dtype=uint8)
# One-Hot Encoding: In one-hot encoding, integer labels are converted into a binary matrix in such a way that:
# Each column in the matrix corresponds to a class label.
# For each instance (row in the matrix), only the column corresponding to its label is set to 1, and all other columns are set to 0.
y_cat_test = to_categorical(y_test,10) # y_test array of integer labels
y_cat_train = to_categorical(y_train,10) # y_cat_train[0] = array([0., 0., 0., 0., 0., 0., 1., 0., 0., 0.], dtype=float32) --> frog

# Processing X Data
# Normalize the X data
# single_image.max() = 255, single_image.min() = 0
x_train = x_train/255 # divided by x_train_max
x_test = x_test/255
scaled_single = x_train[0]
# scaled_single.max() = 1
plt.imshow(scaled_single) # inside jupyter lab
plt.show() # outside jupyter lab

# Data is 60,000 images stored in 32 by 32 pixel array formation with 3 color channels.
# x_train.shape = (50000, 32, 32, 3)
# x_test.shape = (10000, 32, 32, 3)

###########################################################################################################################
# TRAINING THE MODEL

def cifar10_model(x_train, x_test, y_cat_test, y_cat_train):
    ### Create Model ###
    model = Sequential() # Linear stack of layers, where each layer has exactly one input tensor and one output tensor.

    # CONVOLUTIONAL LAYER
    model.add(Conv2D(filters=32, kernel_size=(4,4),input_shape=(32, 32, 3), activation='relu',))
    # 2D convolutional layer to the model
    # filters=32:  filter (or kernel) matrix used to extract features from input images through a process called convolution. Number 32 is often used as a starting point in many CNN architectures, particularly for small to medium-sized datasets.
    # kernel_size=(4,4): Specifies the height and width of the 2D convolution window.
    # input_shape=(32, 32, 3): The shape of the input data: height, width, and channels of the images. In this case, it's set for 32x32 pixel images with 3 color channels (RGB).
    # activation='relu': The activation function used. ReLU (Rectified Linear Unit) is a common activation function in neural networks.

    # POOLING LAYER
    model.add(MaxPool2D(pool_size=(2, 2))) # Reduces the spatial dimensions

    # FLATTEN IMAGES FROM 32 by 32 to 764 BEFORE FINAL LAYER
    model.add(Flatten()) # This layer flattens the multi-dimensional output of the previous layers into a one-dimensional array. This is necessary because fully connected layers (like the Dense layer that follows) expect 1D arrays as input.

    # 256 NEURONS IN DENSE HIDDEN LAYER (YOU CAN CHANGE THIS NUMBER OF NEURONS)
    model.add(Dense(256, activation='relu')) # This adds a fully connected layer with 256 neurons to the network. The relu activation function is used here as well.

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
    model.fit(x_train,y_cat_train,verbose=1,epochs=10) # Can take a while, depending on cpu. 


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
    model.save('cifar10_test.h5') # This model has only a precision of 0.64, therefore improvements would be recommended, since it can lead to false results.  

def larger_cifar10_model(x_train, x_test, y_cat_test, y_cat_train):
     ### Create Model ###
    model = Sequential()

    ## FIRST SET OF LAYERS
    # CONVOLUTIONAL LAYER
    model.add(Conv2D(filters=32, kernel_size=(4,4),input_shape=(32, 32, 3), activation='relu',))
    # CONVOLUTIONAL LAYER
    model.add(Conv2D(filters=32, kernel_size=(4,4),input_shape=(32, 32, 3), activation='relu',))
    # POOLING LAYER
    model.add(MaxPool2D(pool_size=(2, 2)))
    
    ## SECOND SET OF LAYERS
    # CONVOLUTIONAL LAYER
    model.add(Conv2D(filters=64, kernel_size=(4,4),input_shape=(32, 32, 3), activation='relu',))
    # CONVOLUTIONAL LAYER
    model.add(Conv2D(filters=64, kernel_size=(4,4),input_shape=(32, 32, 3), activation='relu',))
    # POOLING LAYER
    model.add(MaxPool2D(pool_size=(2, 2)))

    # FLATTEN IMAGES FROM 32 by 32 to 764 BEFORE FINAL LAYER
    model.add(Flatten())

    # 512 NEURONS IN DENSE HIDDEN LAYER (YOU CAN CHANGE THIS NUMBER OF NEURONS)
    model.add(Dense(512, activation='relu'))

    # LAST LAYER IS THE CLASSIFIER, THUS 10 POSSIBLE CLASSES
    model.add(Dense(10, activation='softmax'))


    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    
    print(model.summary())

    
    ### Train Model ###
    model.fit(x_train,y_cat_train,verbose=1,epochs=20) # Can take a while, depending on cpu. 


    ### Evaluate Model ###
    print(model.metrics_names)
    model.evaluate(x_test,y_cat_test)
    predictions = model.predict_classes(x_test)
    print(classification_report(y_test,predictions))

    ### Save Model ###
    model.save('larger_cifar10_test.h5') # This model has a precision of 0.7. Further improvements can be done. Here 0.7 is sufficient for testing purposes.  


### Run Function to create the model ###
larger_cifar10_model(x_train, x_test, y_cat_test, y_cat_train)