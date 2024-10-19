import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
import cv2

def extract_image_features(data_dir, feature_type):
    """
    Extracts features from a directory of images.

    Args:
        data_dir (str): The directory containing the image folders.
        feature_type (str): The type of features to extract (e.g., 'image').

    Returns:
        pandas.DataFrame: The extracted features.
    """

    if feature_type == 'image':
        # Extract image features
        image_data = []
        labels = []
        for class_label in os.listdir(data_dir):
            class_dir = os.path.join(data_dir, class_label)
            for image_file in os.listdir(class_dir):

                # Exclude hidden files like ".DS_Store"
                if not image_file.startswith('.'):
                    image_path = os.path.join(class_dir, image_file)
                    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    image = cv2.resize(image, (48, 48))
                    image_data.append(image.flatten())
                    labels.append(class_label)
        features = pd.DataFrame(image_data, columns=['pixel_' + str(i) for i in range(48 * 48)])
        features['emotion'] = labels

    else:
        raise ValueError("Invalid feature type. Please choose 'image'.")

    return features

# Example usage
data_dir = "/Users/matthewkim/Desktop/traindataset"  # Replace with the actual path

# Extract image features
image_features = extract_image_features(data_dir, 'image')

# Perform dimensionality reduction if necessary
pca = PCA(n_components=100)  # Adjust n_components as needed
reduced_features = pca.fit_transform(image_features.drop('emotion', axis=1))
reduced_features = pd.DataFrame(reduced_features)

# Combine with original labels
combined_features = pd.concat([reduced_features, image_features['emotion']], axis=1)

# Visualize extracted features (optional)
plt.scatter(reduced_features[0], reduced_features[1])
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("Feature Visualization")
plt.show()