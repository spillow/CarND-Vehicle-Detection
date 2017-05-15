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
from moviepy.editor import VideoFileClip
from IPython.display import HTML

def draw_boxes(img, bboxes, thick=6):
    c = np.max(img)
    c = 255 if c > 1 else 1
    color = (0, 0, c)
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

def color_hist(img, nbins=32):
    # Compute the histogram of the RGB channels separately
    rhist = np.histogram(img[:,:,0], bins=nbins)
    ghist = np.histogram(img[:,:,1], bins=nbins)
    bhist = np.histogram(img[:,:,2], bins=nbins)
    # Generating bin centers
    bin_edges = rhist[1]
    bin_centers = (bin_edges[1:]  + bin_edges[0:len(bin_edges)-1])/2
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((rhist[0], ghist[0], bhist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return rhist, ghist, bhist, bin_centers, hist_features

def color_convert(img, color_space):
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
    else:
        feature_image = np.copy(img)

    return feature_image

def bin_spatial(img, size=(32, 32)):
    features = cv2.resize(img, size).ravel()
    return features

def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    # Use skimage.hog() to get both features and a visualization
    output = hog(
        img,
        orientations=orient,
        pixels_per_cell=(pix_per_cell, pix_per_cell),
        cells_per_block=(cell_per_block, cell_per_block),
        visualise=vis,
        feature_vector=feature_vec,
        transform_sqrt=True)

    return output

def extract_features(img):
    orient = 9
    cell_per_block = 2
    pix_per_cell = 8

    features = []

    for space in ['YCrCb']:
        conv_img = color_convert(img, space)
        hog1 = get_hog_features(conv_img[:,:,0], orient, pix_per_cell, cell_per_block)
        hog2 = get_hog_features(conv_img[:,:,1], orient, pix_per_cell, cell_per_block)
        hog3 = get_hog_features(conv_img[:,:,2], orient, pix_per_cell, cell_per_block)
        (_, _, _, _, hist_features) = color_hist(conv_img)
        spatial_features = bin_spatial(conv_img)

        features += [hog1, hog2, hog3, hist_features, spatial_features]

    features = np.concatenate(features)

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
    window_sizes = [(128, 0.8, 400, 650, 640), (64, 0.8, 400, 650, 640), (80, 0.97, 410, 410+80, 640)]

    all_boxes = []
    candidate_windows = []
    for (win_size, overlap, ystart, ystop, xstart) in window_sizes:
        windows = slide_window(img,
            xy_window=(win_size, win_size),
            xy_overlap=(overlap, overlap),
            y_start_stop=(ystart, ystop),
            x_start_stop=(xstart, None))
        for ((x1, y1), (x2, y2)) in windows:
            all_boxes.append(((x1, y1), (x2, y2)))
            features = extract_features(cv2.resize(img[y1:y2, x1:x2, :], (64, 64)))
            features = scaler.transform(np.array(features).reshape(1, -1))
            if classifier.predict(features) == 1:
                candidate_windows.append(((x1, y1), (x2, y2)))

    return (candidate_windows, all_boxes)

def find_labeled_bboxes(img, labeling, num_cars):
    boxes = []
    for car_number in range(1, num_cars+1):
        nonzero = (labeling == car_number).nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        boxes.append(bbox)

    return boxes

def average_heatmap(heatmap, frame_info):
    if frame_info is None or len(frame_info) == 0:
        return heatmap

    return np.mean([f.heatmap for f in frame_info] + [heatmap], axis=0)

def find_cars(img, classifier, scaler, frame_info):
    (candidate_boxes, all_boxes) = find_car_candidates(img, classifier, scaler)
    #show_img(draw_boxes(np.copy(img), all_boxes))
    #show_img(draw_boxes(np.copy(img), candidate_boxes))
    heatmap = np.zeros(img.shape[:2])
    heatmap = add_heat(heatmap, candidate_boxes)
    #show_img(heatmap, cmap='gray')
    avg_heatmap = average_heatmap(heatmap, frame_info)
    #show_img(avg_heatmap, cmap='gray')
    thresh_heatmap = apply_threshold(avg_heatmap, 1)
    #show_img(thresh_heatmap, cmap='gray')

    (labeling, num_cars) = label(thresh_heatmap)
    #show_img(labeling, cmap='gray')
    bboxes = find_labeled_bboxes(img, labeling, num_cars)
    #show_img(draw_boxes(np.copy(img), bboxes))

    return (bboxes, heatmap)

def sanity_check(boxes):
    # only select boxes of minimum dimensions.
    bs = []
    for (b, ((x1, y1), (x2, y2))) in zip(boxes, boxes):
        if abs(x1-x2) < 10 or abs(y1-y2) < 10:
            continue

        bs.append(b)

    return bs

def normalize_data(*inputs):
    X = np.vstack(inputs).astype(np.float64)
    X_scaler = StandardScaler().fit(X)
    scaled_X = X_scaler.transform(X)

    return (scaled_X, X_scaler)

def show_img(img, cmap=None):
    plt.figure()
    plt.imshow(img, cmap)
    plt.show()

class FrameInfo:
    def __init__(self, boxes, heatmap):
        self.boxes   = boxes
        self.heatmap = heatmap

def rect_format(boxes):
    rects = []
    for ((x1,y1), (x2, y2)) in boxes:
        rects.append((int(x1), int(y1), int(abs(x2-x1)), int(abs(y2-y1))))

    return rects

def inter_frame_analysis(boxes, frame_info):
    if frame_info is None:
        return boxes
    elif len(frame_info) == 0:
        return []

    (rects, _) = cv2.groupRectangles(
        rect_format(boxes) + rect_format(frame_info[-1].boxes),
        groupThreshold=1,
        eps=0.3)

    return [((x, y), (x+w, y+h)) for (x,y,w,h) in rects]

def annotate_image(img, classifier, scaler, frame_info):
    imgcopy = np.copy(img).astype(np.float32)
    imgcopy /= 255

    (boxes, heatmap) = find_cars(imgcopy, classifier, scaler, frame_info)
    boxes = sanity_check(boxes)
    #show_img(draw_boxes(np.copy(img), boxes))
    car_boxes = inter_frame_analysis(boxes, frame_info)

    img = draw_boxes(img, car_boxes)

    return (img, FrameInfo(boxes, heatmap))

def main():
    #img = mpimg.imread('test_images/test1.jpg')
    #windows = slide_window(img, xy_window=(128,128))
    #for (p1, p2) in windows:
    #    cv2.rectangle(img, p1, p2, color=(0,0,255), thickness=6)

    #plt.figure()
    #plt.imshow(img)
    #plt.show()

    #return

    #def dump_video(input, output):
    #    cnt = 0
    #    def process_image(img):
    #        nonlocal cnt
    #        mpimg.imsave("clean_output/{}.jpg".format(cnt), img)
    #        cnt += 1
    #        return img
    #    clip = VideoFileClip(input)
    #    clip_output = clip.fl_image(process_image)
    #    clip_output.write_videofile(output, audio=False)

    #dump_video('project_video.mp4', 'project_ident.mp4')
    #return

    car_features    = train_extract_features(glob.iglob('training_data/vehicles/**/*.png'))
    notcar_features = train_extract_features(glob.iglob('training_data/non-vehicles/**/*.png'))
    #car_features    = train_extract_features(glob.iglob('garbage_test/*.png'))
    #notcar_features = train_extract_features(glob.iglob('garbage_test/*.png'))

    (X_data, scaler) = normalize_data(car_features, notcar_features)
    Y_data           = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    classifier = train_classifer(X_data, Y_data)

    def process_video(input, output):
        frame_info = []
        def process_image(img):
            nonlocal frame_info
            (img, frameinfo) = annotate_image(img, classifier, scaler, frame_info)
            frame_info.append(frameinfo)
            if len(frame_info) > 4:
                frame_info.pop(0)
            return img
        clip = VideoFileClip(input)
        clip_output = clip.fl_image(process_image)
        clip_output.write_videofile(output, audio=False)

    #process_video('test_video.mp4', 'output.mp4')
    #process_video('project_video.mp4', 'output_full.mp4')
    process_video('tough_case.mp4', 'tough_case_anno.mp4')

    #for path in glob.iglob('test_images/6*.jpg'):
    #    img = mpimg.imread(path)
    #    show_img(annotate_image(img, classifier, scaler, None)[0])

if __name__ == '__main__':
    main()

