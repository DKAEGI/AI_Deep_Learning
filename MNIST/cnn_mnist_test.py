from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import cv2

# Load the model
model = load_model('mnist_test.h5')

# Load and preprocess the image
def preprocess_image(image_path, target_size):
    # Load image in grayscale mode
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
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


# Assuming model expects 28x28 pixel images
image = preprocess_image('images_test/handwritten_digit_9.png', target_size=(28, 28))

prediction = model.predict(image)

print(prediction)

predicted_class = np.argmax(prediction, axis=1) # Highest probability
print(f'Predicted class: {predicted_class[0]}')
