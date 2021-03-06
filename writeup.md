
**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Additionally, a color transform and binned color features, as well as histograms of color, were added to the HOG features to boost classifier accuracy.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG.png
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/candidate_boxes.png
[image5]: ./examples/final_boxes.png
[image6]: ./examples/combo_heat.png
[image7]: ./examples/labeled.png
[image8]: ./examples/final_boxer.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points

---

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/spillow/CarND-Vehicle-Detection/blob/master/writeup.md) is the link.

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained on line #85 of the file called `car_pipeline.py` in the function `get_hog_features()`.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  The starting values of `orientations = 9`, `pixels_per_cell = 8`, and `cells_per_block = 2` was ultimately modified to `orientations = 11`, `pixels_per_cell = 16`, and `cells_per_block = 2`. I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(16, 16)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

While the initial and final setting of values both yielded classifiers of roughly 99% test accuracy, the latter numbers yieled a 33% performance improvement which was a big win with the slow processing times of this project.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features, color features, and spatial features.

I trained a linear SVM using sklearn in the function `train_classifier()` on line #126.  I used a GridSearch approach as
the best `C` parameter to control the complexity of the svm decision boundary was not obvious.  Before this point, the HOG, color, and spatial features were extracted from each of the training images, combined into one long feature vector in `extract_features()` on line #98, and normalized.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I implemented a sliding window search in `slide_window()` on line #25. Given a region, window size, and amount of overlap, the function emits a list of windows.

Scales were initially selected such that they would roughly match the sizes of the cars of interest.  I noted that, due to the fact that the training data only had cars with minimal offset from being centered, the training windows needed to be fairly well targeted to the car.  Another set of windows targeted to the road horizon was added with 0.97 overlap to capture cars that I was having difficulty detecting with only the two sets of window sizes (32x32 and 64x64, respectively).

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

It was important to limit the region of the image to search (e.g., don't look in the sky).  This helps performance in the timing sense as well as the accuracy sense.

I also lightly experimented with the recursive feature elimination in sklearn in an attempt to lower the feature count.  No progress that but worth further exploration.

![alt text][image4]
![alt text][image5]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](https://github.com/spillow/CarND-Vehicle-Detection/blob/master/output.mp4)

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

False positives were mostly controlled via heatmaps which are dealt with in the functions `add_heat()`, `apply_threshold()`, and `average_heatmap()`.  While heatmapping and thresholding eliminated many false positives, I also found it useful to average the heatmaps over 5 frames which would suppress single frames blips from the classifier.

To coalesce multiple hits into a single detection, `scipy.ndimage.measurements.label()` was used.  After each pixel in the heat map was labeled, I then drew a box around the extent of the entire label assuming that was a car.

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on a frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image6]

### Here is the output of `scipy.ndimage.measurements.label()` on one of the frames:
![alt text][image7]

### Here the resulting bounding boxes are drawn onto one of the frames in the series:
![alt text][image8]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

As always, the initial work goes into getting the classifier accuracy to a good enough level.  Being a detector, 99% really is about the minimum that can be tolerated.  We're lucky here to have the redundancy of video to help weed out false positives.

I had an issue with a brief false positive blip along the right guard rail at one point.  I tried mining a collection of examples from that region to teach the classifier that those weren't cars but was unsuccessful.  Perhaps using the larger dataset would help weed out those last cases.

It would also be useful to have the pipline identify cars going the opposite way to, say, avoid cars crossing the double yellow.  The initial dataset doesn't have any of those examples so it is not capable but would be a straightforward extension.

During the portion of the video where one car overtakes the other, they temporarily merge.  It would be useful to teach it the notion of "object permanence" so that it knows another car is back there.

Lastly, this implementation is nowhere close to real time.  I was able to get it to process a frame every 2 seconds.  This is why feature elimination is useful to explore here.  I imagine a convnet approach on a GPU is probably the more state of the art way of going about this for time performance and accuracy.
