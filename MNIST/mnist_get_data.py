from keras.datasets import mnist # Import dataset with handwritten letters
import matplotlib.pyplot as plt

# load the data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Save 10 images and their labels
for i in range(10):
    # Define the file path
    file_path = f'image_{i}_label_{y_train[i]}.png'
    # Save the image
    plt.imsave(file_path, x_train[i], cmap='gray')
    print(f'Saved {file_path}')

