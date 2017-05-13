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
from scipy.ndimage.measurements import label

def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    for (p1, p2) in bboxes:
        cv2.rectangle(img, p1, p2, color, thick)

    return img

def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    (height, width) = img.shape[:2]
    (xstart, ystart) = (x_start_stop[0] if x_start_stop[0] is not None else 0,
                        y_start_stop[0] if y_start_stop[0] is not None else 0)

    (xend, yend) = (x_start_stop[1] if x_start_stop[1] is not None else width-1,
                    y_start_stop[1] if y_start_stop[1] is not None else height-1)
    # Compute the span of the region to be searched
    # Compute the number of pixels per step in x/y
    xstep = int(xy_window[0] - xy_overlap[0] * xy_window[0])
    ystep = int(xy_window[1] - xy_overlap[1] * xy_window[1])
    # Compute the number of windows in x/y
    #xstart + i*xstep + xy_window[0] - 1 <= xend
    # ==>
    xwindows = (xend + 1 - xy_window[0] - xstart) // xstep + 1
    ywindows = (yend + 1 - xy_window[1] - ystart) // ystep + 1
    window_list = []
    for j in range(ywindows):
        for i in range(xwindows):
            window_list.append(((xstart + i*xstep, ystart + j*ystep),
                                (xstart + i*xstep + xy_window[0],
                                 ystart + j*ystep + xy_window[1])))
    return window_list

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

def extract_features(img):
    orient = 9
    cell_per_block = 2
    pix_per_cell = 8
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    features = get_hog_features(gray, orient, pix_per_cell, cell_per_block)

    return features

def train_extract_features(filenames):
    features = []
    for file in filenames:
        img = mpimg.imread(file)
        features.append(extract_features(img))

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

def add_heat(heatmap, car_candidates):
    for ((x1, y1), (x2, y2)) in car_candidates:
        heatmap[y1:y2, x1:x2] += 1

    return heatmap

def apply_threshold(heatmap, threshold):
    heatmap[heatmap <= threshold] = 0
    return heatmap

def find_car_candidates(img, classifier, scaler):
    window_sizes = [128, 64]

    candidate_windows = []
    for win_size in window_sizes:
        windows = slide_window(img,
            xy_window=(win_size, win_size),
            xy_overlap=(0.8, 0.8),
            y_start_stop=(350, None))
        for ((x1, y1), (x2, y2)) in windows:
            features = extract_features(cv2.resize(img[y1:y2, x1:x2, :], (64, 64)))
            features = scaler.transform(np.array(features).reshape(1, -1))
            if classifier.predict(features) == 1:
                candidate_windows.append(((x1, y1), (x2, y2)))

    return candidate_windows

def find_labeled_bboxes(img, labeling, num_cars):
    boxes = []
    for car_number in range(1, num_cars+1):
        nonzero = (labeling == car_number).nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        boxes.append(bbox)

    return boxes

def find_cars(img, classifier, scaler):
    candidate_boxes = find_car_candidates(img, classifier, scaler)
    print(candidate_boxes)
    heatmap = np.zeros(img.shape[:2])
    heatmap = add_heat(heatmap, candidate_boxes)
    show_img(heatmap, cmap='gray')
    heatmap = apply_threshold(heatmap, 2)
    show_img(heatmap, cmap='gray')

    (labeling, num_cars) = label(heatmap)
    show_img(labeling, cmap='gray')
    bboxes = find_labeled_bboxes(img, labeling, num_cars)

    return bboxes

def normalize_data(*inputs):
    X = np.vstack(inputs).astype(np.float64)
    X_scaler = StandardScaler().fit(X)
    scaled_X = X_scaler.transform(X)

    return (scaled_X, X_scaler)

def show_img(img, cmap=None):
    plt.figure()
    plt.imshow(img, cmap)
    plt.show()

def main():
    #img = mpimg.imread('test_images/test1.jpg')
    #windows = slide_window(img, xy_window=(128,128))
    #for (p1, p2) in windows:
    #    cv2.rectangle(img, p1, p2, color=(0,0,255), thickness=6)

    #plt.figure()
    #plt.imshow(img)
    #plt.show()

    #return

    car_features    = train_extract_features(glob.iglob('training_data/vehicles/**/*.png'))
    notcar_features = train_extract_features(glob.iglob('training_data/non-vehicles/**/*.png'))
    #car_features    = train_extract_features(glob.iglob('garbage_test/*.png'))
    #notcar_features = train_extract_features(glob.iglob('garbage_test/*.png'))

    (X_data, scaler) = normalize_data(car_features, notcar_features)
    Y_data           = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    classifier = train_classifer(X_data, Y_data)

    img = mpimg.imread('test_images/test1.jpg')

    imgcopy = np.copy(img)
    imgcopy = imgcopy.astype(np.float32)
    imgcopy /= 255

    boxes = find_cars(imgcopy, classifier, scaler)

    draw_boxes(img, boxes)

    show_img(img)

if __name__ == '__main__':
    main()

