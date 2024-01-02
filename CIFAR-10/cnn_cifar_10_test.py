from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import cv2

# Load the model
model = load_model('larger_cifar10_test.h5')

# Load and preprocess the image
def preprocess_image(image_path, target_size):
    # Load image in grayscale mode
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Check if the image was correctly loaded
    if img is None:
        raise ValueError(f"Image at {image_path} cannot be loaded")
    
    # Resize the image to target_size with interpolation for better quality
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_CUBIC)
    
    # Normalize the image pixels to be in the range [0, 1]
    img = img / 255.0
    
    # Convert the image to a numpy array
    img = img_to_array(img)
    
    # Add a batch dimension (model expects a batch of images, not a single image)
    img = np.expand_dims(img, axis=0)
    
    return img


# Assuming model expects 32x32 pixel images
image = preprocess_image('images_test/dog.jpg', target_size=(32, 32))

prediction = model.predict(image)

print(prediction)

predicted_class = np.argmax(prediction, axis=1) # Highest probability

# Define a dictionary representing the hash map
class_labels = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck"
}

# Accessing values using keys
key = predicted_class[0]

if key in class_labels:
   label = class_labels[key]
   print(f"The class label for key {key} is {label}")
else:
   print(f"No class label found for key {key}")

