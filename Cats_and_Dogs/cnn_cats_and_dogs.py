import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D
from sklearn.metrics import classification_report
import cv2
from keras.preprocessing.image import ImageDataGenerator

import warnings
warnings.filterwarnings('ignore')

### Visualize data ###
path = 'C:/Users/Dkaeg/Documents/ROS/udemy_python/Python for Computer Vision/Computer-Vision-with-Python/DATA/CATS_DOGS/CATS_DOGS'
image = '/train/CAT/4.jpg'    
cat4 = cv2.imread(path+image)
cat4 = cv2.cvtColor(cat4,cv2.COLOR_BGR2RGB)
plt.imshow(cat4)
plt.show()

image = '/train/DOG/2.jpg'    
dog2 = cv2.imread(path+image)
dog2 = cv2.cvtColor(dog2,cv2.COLOR_BGR2RGB)
plt.imshow(dog2)
plt.show()

### Image Manipulation ###
# Its usually a good idea to manipulate the images with rotation, resizing, and scaling so the model becomes more robust to different images that our data set doesn't have. Use the ImageDataGenerator to do this automatically.
image_gen = ImageDataGenerator(rotation_range=30, # rotate the image 30 degrees
                               width_shift_range=0.1, # Shift the pic width by a max of 10%
                               height_shift_range=0.1, # Shift the pic height by a max of 10%
                               rescale=1/255, # Rescale the image by normalzing it.
                               shear_range=0.2, # Shear means cutting away part of the image (max 20%)
                               zoom_range=0.2, # Zoom in by 20% max
                               horizontal_flip=True, # Allo horizontal flipping
                               fill_mode='nearest' # Fill in missing pixels with the nearest filled value
                              )
plt.imshow(image_gen.random_transform(dog2))
plt.show()

# Generating many manipulated images from a directory
# In order to use .flow_from_directory, the images must be organized in sub-directories. This is an absolute requirement, otherwise the method won't work. The directories should only contain images of one class, so one folder per class of images.
image_gen.flow_from_directory(path + '/train')
image_gen.flow_from_directory(path + '/test')

# Resizing Images, Let's have Keras resize all the images to 150 pixels by 150 pixels once they've been manipulated.
# width,height,channels
image_shape = (150,150,3)

###########################################################################################################################
# TRAINING THE MODEL

def create_model():
    ### Create Model ###
    model = Sequential()

    model.add(Conv2D(filters=32, kernel_size=(3,3),input_shape=(150,150,3), activation='relu',))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=64, kernel_size=(3,3),input_shape=(150,150,3), activation='relu',))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=64, kernel_size=(3,3),input_shape=(150,150,3), activation='relu',))
    model.add(MaxPooling2D(pool_size=(2, 2)))


    model.add(Flatten())


    model.add(Dense(128))
    model.add(Activation('relu'))

    # Dropouts help reduce overfitting by randomly turning neurons off during training.
    # Here we say randomly turn off 50% of neurons.
    model.add(Dropout(0.5))

    # Last layer, remember its binary, 0=cat , 1=dog
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    print(model.summary())


    ### Train Model ###
    
    # Process 16 images at a time during each training step
    batch_size = 16 
    
    # Data augmentation
    train_image_gen = image_gen.flow_from_directory(path + '/train',
                                                   target_size=image_shape[:2],
                                                   batch_size=batch_size,
                                                   class_mode='binary')
    
    test_image_gen = image_gen.flow_from_directory(path + '/test',
                                               target_size=image_shape[:2],
                                               batch_size=batch_size,
                                               class_mode='binary')
    
    # Print which class is assigned index 0 and which is assigned index 1.
    print(train_image_gen.class_indices)
    
    # 150 batches per training step and 12 batches per validation step
    results = model.fit_generator(train_image_gen,epochs=100,
                              steps_per_epoch=150,
                              validation_data=test_image_gen,
                             validation_steps=12) 

    ### Evaluate Model ###
    plt.plot(results.history['acc'])

    
    ### Save Model ###
    model.save('cats_and_dogs.h5')

create_model()