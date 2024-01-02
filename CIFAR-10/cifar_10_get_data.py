from keras.datasets import cifar10
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Save 10 images and their labels
for i in range(10):
    # Define the file path
    file_path = f'images_test/image_{i}_label_{y_train[i]}.png'
    # Save the image
    plt.imsave(file_path, x_train[i], cmap='gray')
    print(f'Saved {file_path}')

