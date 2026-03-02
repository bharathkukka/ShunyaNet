# Image Feature Extraction using Higher Level Representations
# This code demonstrates how to extract higher-level features from images using a pre-trained convolutional neural network (CNN) model.
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input


from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
import os
# Load the pre-trained VGG16 model
base_model = VGG16(weights='imagenet')
# Create a new model that outputs features from the 'fc1' layer
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)
# Function to extract higher-level features from an image
def extract_higher_level_features(img_path):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    # Extract features
    features = model.predict(x)
    return features
# Example usage
if __name__ == "__main__":
    # Path to the image
    img_path = '/Users/bharathgoud/PycharmProjects/CNN/Lab/diwali.jpeg'  # Replace with your image path
    # Extract features
    features = extract_higher_level_features(img_path)
    print("Extracted Features Shape:", features.shape)
    print("Extracted Features:", features)
    # Visualize the original image
    img = image.load_img(img_path, target_size=(224, 224))
    plt.imshow(img)
    plt.axis('off')
    plt.show()


    # Optionally, save the features to a file
    np.save('extracted_features.npy', features)
# Save the features to a file
    np.save('extracted_features.npy', features)

    print("Features saved to 'extracted_features.npy'")

