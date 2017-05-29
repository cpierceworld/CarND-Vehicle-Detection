# **Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

Link to my [project code](https://github.com/cpierceworld/CarND-Vehicle-Detection/blob/master/Vehicle_Detection.ipynb)

[//]: # (Image References)
[image1]: ./output_images/example_cars.png
[image2]: ./output_images/example_non_cars.png
[image3]: ./output_images/hog_car.png
[image4]: ./output_images/hog_non_car.png
[image5]: ./output_images/color_hist_car.png
[image6]: ./output_images/color_hist_non_car.png
[image7]: ./output_images/color_spatial_car.png
[image8]: ./output_images/color_spatial_non_car.png
[image9]: ./output_images/search_windows.png
[image10]: ./output_images/find_cars_raw.png
[image11]: ./output_images/heatmap_frames.png
[image12]: ./output_images/heatmap_avg.png
[video1]: ./project_video_out.mp4


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/513/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf. 

A copy of this writeup can be found [here](https://github.com/cpierceworld/CarND-Vehicle-Detection/blob/master/writeup.md).  The code this writeup refers to can be found [here](https://github.com/cpierceworld/CarND-Vehicle-Detection/blob/master/Vehicle_Detection.ipynb)

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

I wrote a class to contain all classification logic called `VehicleClassifier`, which is in 10th code cell of the Jupyter Notebook.  The Classifier uses 3 feature sets:
1. HOG
2. Histograms of Color
3. Spatial Binning of Color

All 3 feature set's hyperparameters are configurable via the `VehicleClassifier.config` map, including the ability to turn a particular feature set on and off.

I explored each feature set using different color spaces and hyperparameters on an example "Car" and "Non-Car" image.

##### _HOG Features_

The code for generating HOG features is in 5th code cell of the notebook.  I utilized the [_sklearn_](http://scikit-learn.org/stable/index.html) implementation of HOG.

I explored different color spaces and different `skimage.feature.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image3]
![alt text][image4]

##### _Histogram of Color Features_

In addition to HOG, I used Histogram of Color features for my classifier.  The code can be found in the 7th code cell in the notebook. I experimented with having it and not having it and found that classifcation worked better with it.  I experimented with various color spaces and number of bins.   Here is an example histogram using the `YCrCb` color space with 32 bins per channel:

![alt text][image5]
![alt text][image6]

##### _Spatial Binning of Color Features_

I also used Spatial Binning of Color features for my classifier.  The code can be found in the 7th code cell in the notebook. Again I experimented with having it and not having it and found that classification worked better with it.   I experimented with various color spaces and image sizes.  Here is an example of using `YCrCb` color space, resized to 32x32:

![alt text][image7]
![alt text][image8]

#### _Scaling the Feature Vector_

The resulting feature vector is the concatenation of the HOG, Histogram, and Binned Color features which have values in different scales (e.g. "0.0 - 1.0" vs. "0 - 3000" vs. "0 - 255").  In order to not have one feature set dominate just due to the scale of it's output values, the code rescales the feature vector using `sklearn.preprocessing.StandardScaler`.  (This is done in the `VehicleClassifier.train()` method, after stakcing together all the feature vectors).

#### 2. Explain how you settled on your final choice of HOG parameters.

I found that computing the HOG features took a long time with a small `pixels_per_cell` value.  Doubling the `pixels_per_cell` resulted in predections taking only a third of the time, however this caused lower accuracy.  By increasing the number of `orientations`, I could mitigate the lower accuracy without a significant impact on pediction speed.

The final HOG hyperparameters was `orientations=12` and `pixels_per_cell=(16,16)`.

(**Note**: I acutally wrote a loop to test various combinations of hyperparemters, see section 3 below).

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I started by reading in all the `vehicle` and `non-vehicle` images.  Here are examples of the `vehicle` and `non-vehicle` classes:

![alt text][image1]
![alt text][image2]

I performed a "parameter search" in code block 13 of the Jupyter notebook.  It loops over various combinations of SVM parameters (`kernel`, `C`) and HOG (`orientations`, `pixels_per_cell`).

My goal was to find a setting that was sufficiently accurate, but took no longer than half a second to make a prediction.  Below is the table of the results:

| Orient | Pix/Cell | C        | Kernel   | Accuracy | Prediction Time (s) | 
|:------:|:--------:|:--------:|:--------:|:--------:|:-------------------:| 
| 12     | 16       | 1        | rbf      | 0.9955   | 1.59                | 
| 8      | 16       | 1        | rbf      | 0.9944   | 1.63                | 
| 12     | 8        | 0.5      | linear   | 0.9941   | 0.61                | 
| 10     | 16       | 0.5      | rbf      | 0.9933   | 1.53                | 
| 12     | 16       | 1        | linear   | 0.9924   | 0.19                | 
| 8      | 8        | 1        | linear   | 0.9921   | 0.51                | 
| 8      | 8        | 0.5      | linear   | 0.9907   | 0.53                | 
| 10     | 8        | 1        | linear   | 0.9907   | 0.55                | 
| 12     | 16       | 1        | poly     | 0.9907   | 3.52                | 
| 8      | 16       | 0.5      | linear   | 0.9904   | 0.16                | 
| 12     | 8        | 2        | linear   | 0.9904   | 0.58                | 
| 12     | 16       | 0.5      | linear   | 0.9904   | 0.19                | 
| 10     | 16       | 2        | linear   | 0.9902   | 0.17                | 
| 12     | 16       | 2        | linear   | 0.9902   | 0.18                | 
| 8      | 16       | 2        | linear   | 0.9899   | 0.16                | 
| 12     | 8        | 1        | linear   | 0.9899   | 0.62                | 
| 10     | 16       | 1        | poly     | 0.9899   | 3.32                | 
| 8      | 16       | 1        | linear   | 0.9893   | 0.16                | 
| 10     | 16       | 1        | linear   | 0.9893   | 0.17                | 
| 8      | 8        | 2        | linear   | 0.9888   | 0.51                | 
| 10     | 8        | 0.5      | linear   | 0.9888   | 0.61                | 
| 10     | 8        | 2        | linear   | 0.9885   | 0.56                | 
| 10     | 16       | 0.5      | linear   | 0.9876   | 0.16                | 
| 8      | 16       | 1        | poly     | 0.9831   | 3.16                | 

I ended up using `orientations=12`, `pixels_per_cell=(16,16)`, `C=1`, `kernel=linear` as hyperparameters which had an accuracy 99.24% and a prediction time of only 0.19 seconds.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

My approach was to use "small/medium/large" sliding windows with the smaller windows only at the horizon where cars would be distant, medium windows to search the middle region, and large windows near the bottom where close (large) cars would be found.  

Below is an image showing the sliding windows.

![alt text][image9]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Below is the classifier processing 6 example images:

![alt text][image10]

I made various attemtps to optimize the performance:
1. I tried using _OpenCV_'s HOG implementation with _CUDA_ enabled (code can still be found in the 5th code cell).   It didn't really improve the speed by any significant amount, and accuracy went down so I went back to _sklearn_.
2. I changed the HOG parameters to use 16 pixels per cell vs. 8 which which trippled the prediction speed (see section 3 about choosing SVM/HOG parameters with regards to performance)
3. I used the "HOG Sub-Sampling" technique rather than the sliding windows to calculate all the HOG features at once.  Code is in the `VehicleClassier` class in the `find_cars()` method.
4. I played with the number of sliding windws, originally starting with 5 sizes searching the entire lower half of the image, which resulted in minutes per frame to processes.  I settled on 3 sizes in more targeted regions.

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](./project_video_out.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I wrote a class called `CarTracker` to contain the logic of executing the search and proccessing the ouput of the `VehicleClassifier`.  This code can be found in 11th code cell in the Jupyter notebook.

This class is responsible for:
1. Doing the "sliding window" search.
2. Filtering out false positives.
3. Tracking individual cars across frames.

The `CarTracker.find_cars()` method does all processing; the logic goes like this:

1. **Sliding Window**: It does the "sliding window" search using the `VehicleClassifier` which returns many (overlapping) potential dections (true and false detections).

2. **Heatmap**: Creates a "heatmap" from the potential detections (overlapping detections results in "more heat").  The assumption is that false positives will have less overlap then true detections.

3. **Heatmap Averaging**: Uses "heatmap averaging" to further dilute false positives.  The code keeps a history of heatmaps over a configurable number of video frames (4 by default). Assuming that false positives are not likely accross multiple frames of video, it calculates a "heatmap average" which reduces the values in false positive regions.

4. **Heatmap Threshold**: Thresholds the heatmap, removing regions of "low heat".  These regions should correspend with false positives.

5. **Reheat**: This step is to counteract the phenomena where thresholding tends to shrink the region around a true detection since the fringe area has low heat (resulting in a bounding box that doesn't encapsulate the car).   This step adds heat back to any detection that overlaps a region that still has "heat" after the thresholding from step 4.

5. **Label**: Uses `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap, which are assumed to corresponded to vehicles. Bounding boxes are constructed to cover the area of each blob detected.

6. **Car Reconciliation**: This step is to reconcile the (potential) vehicles found in the current frame with vehicles found in the previous frame(s).   Bounding boxes that overlap a previously found vehicle are assumed to be the same vehicle.   A new bounding box is calculated as the average of the previous and new.    Bounding boxes that don't overlap any previously found vehicles are assumed to be new vehicles.  If there is a previously found vehicle that doesn't overlap any current bounding box, then that vehicle's "not detected" count is incremented (see step 8 "Max Frame No-Detect Filtering").

7. **Min Frame Detect Filtering**:  As a last attempt to filter out false positives, the code has a configurable parameter for "min frame detect filtering".   This is the minimum number of consecutive frames that a car must be detected in before it is recognized as a "Car", given an ID and plotted.  The assumption is that false positives won't be detected in consecutive frames.

8. **Max Frame No-Detect Filtering**: In an attempt to not loose track of cars when they are mistakenly not detected, there is "max frame no-detect filtering".  This is the maximum number of consecutive frames that a car can not be detected before the tracker stops trying to track it.


### Here are four frames and their corresponding heatmaps:

![alt text][image11]

### Here is the result of "Heatmap Averaging" over the four frames and the detected blobs from the final image:
![alt text][image12]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Performance was an issue throughout this project.  Long processing times limited the amount of experimentation.  The final product is still far too slow for any real time processing, averaging only 1 and a half frames per second.   Thoughts to speed it up include switching to OpenCV's SVM implementation which may be faster (assuming it utilzies the GPU).   My other idea was to try switching from `LinearSVC()` to a simple neural net (1 fully connected layer and a softmax) to process the feature vector and see how accurate that is (can use TensorFlow with GPU enabled).

I was significantly delayed by a bad assumption that _sklearn_'s `SVC(kernel='linear')` was equivalent to `LinearSVC()`.   My early attempt at this project utilized `LinearSVC()` and was getting close to 2 frames per second processing.   I then tried to "improve" my project with automatic hyperperameter tuning, including the SVM kernel, so I switched to using the `SVC()` class that takes the kernel as a parameter.  I found that my "auto-tuned" implementation was getting less than a frame per second, and switching back to my original params made no difference.   After may hours of playing with parameters to try and figure out the cause of the slowdown, I finally found that it was the switch from `LinearSVC()` to `SVC(kernel='linear')`.   In the `VehicleClassification.train()` method, I added logic to see if the user passed in a kernel of 'linear' and create a `LinearSVC()` in that scenario.

If I were to do this project again, I don't think I would use "HOG"/"SVM".  I would like to experiment with more recent solutions for "object detection with localization" such as ["You Only Look Once (YOLO)"](https://github.com/pjreddie/darknet/wiki/YOLO:-Real-Time-Object-Detection) and ["Single Shot Detection (SSD)"](https://github.com/weiliu89/caffe/tree/ssd).
