import os

import cv2
import numpy as np
from PIL import Image
from skimage import io
from skimage.color import rgb2gray
from skimage.feature import local_binary_pattern
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def get_mean_color(image_path):
    # Load the image
    img = Image.open(image_path)

    # Extract the mean RGB values
    mean_color = img.convert('RGB').resize((1, 1)).getpixel((0, 0))

    return mean_color


def get_lbp_feature(image_path, radius=3, n_points=8):
    # Load the image and convert to grayscale
    img = rgb2gray(io.imread(image_path))

    # Convert the image to uint8
    img_int = (img * 255).astype(np.uint8)

    # Extract the LBP feature
    lbp = local_binary_pattern(img_int, n_points, radius, method='uniform')

    # Calculate the histogram of LBP codes
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(
        0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype('float')
    hist /= (hist.sum() + 1e-7)

    return hist


def get_hough_lines(image_path, rho=1, theta=np.pi/180, threshold=100):
    # Load the image and convert to grayscale
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Apply the Hough transform to detect lines
    lines = cv2.HoughLines(edges, rho, theta, threshold)

    if lines is None:
        return 0
    else:
        return len(lines)


def main():

    # TRAINING
    data_path = './Forest_Fire_Dataset/Training'
    labels = ["fire", "nofire"]

    output_file = './results.txt'

    directory_list = []

    # Define the number of top features to select
    k = 14

    # Define the dataset and labels
    X = []
    y = []
    color_list = []
    hist_list = []
    lines_list = []
    for label in labels:
        label_dir = os.path.join(data_path, label)
        for filename in os.listdir(label_dir):
            image_path = os.path.join(label_dir, filename)
            color = get_mean_color(image_path)
            hist = get_lbp_feature(image_path)
            lines = get_hough_lines(image_path)
            features = np.concatenate([color, hist, [lines]])
            color_list.append(color)
            hist_list.append(hist)
            lines_list.append(lines)
            X.append(features)
            y.append(label)
        with open("results.txt", "a") as f:
            f.write("Statistics for " + label + " group:\n")
            f.write("Mean color: " + str(np.mean(color_list, axis=0)) + "\n")
            f.write("Mean LBP feature: " +
                    str(np.mean(hist_list, axis=0)) + "\n")
            f.write("Mean Hough lines: " + str(np.mean(lines_list)) + "\n")

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=100)

    # Normalize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Select the top k features
    selector = SelectKBest(f_classif, k=k)
    X = selector.fit_transform(X, y)

    # Define the classifier
    clf = svm.SVC(kernel='linear')

    clf.fit(X_train, y_train)

    # TESTING
    # Define the path to the test data
    test_path = './Forest_Fire_Dataset/Testing'

    # Define the dataset and labels for the test data
    X_test = []
    y_test = []
    for label in labels:
        label_dir = os.path.join(test_path, label)
        for filename in os.listdir(label_dir):
            image_path = os.path.join(label_dir, filename)
            color = get_mean_color(image_path)
            hist = get_lbp_feature(image_path)
            lines = get_hough_lines(image_path)
            features = np.concatenate([color, hist, [lines]])
            X_test.append(features)
            y_test.append(label)

    # Normalize the test features using the same scaler object
    X_test = scaler.transform(X_test)

    # Apply the same feature selection using the same selector object
    X_test = selector.transform(X_test)

    # Use the trained classifier to predict the class labels for the test data
    y_pred = clf.predict(X_test)

    # Evaluate the accuracy of the model on the test data
    accuracy = accuracy_score(y_test, y_pred)
    f1_scores = f1_score(y_test, y_pred, average='weighted', labels=labels)
    with open("results.txt", "a") as f:
        f.write("Accuracy: " + str(accuracy) + "\n")
        f.write("F1 Scores: " + str(f1_scores) + "\n")


if __name__ == "__main__":
    main()
