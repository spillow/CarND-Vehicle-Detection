import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
import glob
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    imcopy = np.copy(img)
    for bbox in bboxes:
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    return imcopy

def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the RGB channels separately
    rhist = np.histogram(image[:,:,0], bins=nbins, range=bins_range)
    ghist = np.histogram(image[:,:,1], bins=nbins, range=bins_range)
    bhist = np.histogram(image[:,:,2], bins=nbins, range=bins_range)
    # Generating bin centers
    bin_edges = rhist[1]
    bin_centers = (bin_edges[1:]  + bin_edges[0:len(bin_edges)-1])/2
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((rhist[0], ghist[0], bhist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return rhist, ghist, bhist, bin_centers, hist_features

def bin_spatial(img, color_space='RGB', size=(32, 32)):
    # Convert image to new color space (if specified)
    # Use cv2.resize().ravel() to create the feature vector
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(img)
    features = cv2.resize(feature_image, size).ravel() # Remove this line!
    # Return the feature vector
    return features

def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    # Use skimage.hog() to get both features and a visualization
    # TODO: transform_sqrt?
    output = hog(
        img,
        orientations=orient,
        pixels_per_cell=(pix_per_cell, pix_per_cell),
        cells_per_block=(cell_per_block, cell_per_block),
        visualise=vis,
        feature_vector=feature_vec)

    return output

def extract_features(filenames):
    features = []
    orient = 9
    cell_per_block = 2
    pix_per_cell = 8
    for file in filenames:
        img = mpimg.imread(file)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        features.append(get_hog_features(gray, orient, pix_per_cell, cell_per_block))

    return features

def train_classifer(X_data, Y_data):
    rand_state = np.random.randint(0, 100)
    X_train, X_test, Y_train, Y_test = train_test_split(
            X_data, Y_data, test_size=0.2, random_state=rand_state)

    parameters = { 'C' : [1, 10] }
    svc = LinearSVC()
    clf = GridSearchCV(svc, parameters)

    print("Training...")

    clf.fit(X_train, Y_train)

    print('Test Accuracy of SVM =', clf.score(X_test, Y_test))

    return clf

def normalize_data(*inputs):
    X = np.vstack(inputs).astype(np.float64)
    X_scaler = StandardScaler().fit(X)
    scaled_X = X_scaler.transform(X)

    return scaled_X

def main():
    car_features    = extract_features(glob.iglob('training_data/vehicles/**/*.png'))
    notcar_features = extract_features(glob.iglob('training_data/non-vehicles/**/*.png'))

    X_data = normalize_data(car_features, notcar_features)
    Y_data = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    classifier = train_classifer(X_data, Y_data)

if __name__ == '__main__':
    main()

