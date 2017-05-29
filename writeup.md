# **Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

Link to my [project code](https://github.com/cpierceworld/CarND-Vehicle-Detection/blob/wip/Vehicle_Detection.ipynb)

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

A copy of this writeup can be found [here](https://github.com/cpierceworld/CarND-Vehicle-Detection/blob/wip/writeup.md).  The code this writeup references can be found [here](https://github.com/cpierceworld/CarND-Vehicle-Detection/blob/wip/Vehicle_Detection.ipynb)

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

I wrote a class to contain all classification logic called `VehicleClassifier`, which is in 10th code cell of the Jupyter Notebook.  The Classifier uses 3 feature sets:
1. HOG
2. Histograms of Color
3. Spatial Binning of Color

All 3 feature set's hyper parameters are configurable via the `VehicleClassifier.config` map, including the ability to turn a particular feature set on and off.

I explored each feature set using different color spaces and hyperparemters on an example "Car" and "Non-Car" image.

##### 1. HOG Features

The code for generating HOG features is in 5th code cell of the notebook.  I utilized the [_sklearn_](http://scikit-learn.org/stable/index.html) implementation of HOG.  I also did tests using _OpenCV_'s HOG implementation to see if it would be faster, but it wasn't and it didn't seem to perform as well, so I stuck with _sklearn_.

I explored different color spaces and different `skimage.feature.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image3]
![alt text][image4]

##### 2. Histogram of Color Features

In addition to HOG, I used Histogram of Color features for my classifier.  I did tests with both having it on and off and found that classifcation worked better with it.  I experimented with various color spaces and number of bins.   Here is an example histogram using the `YCrCb` color space with 32 bins per channel:

![alt text][image5]
![alt text][image6]

##### 3. Spatial Binning of Color Features

I also used Spatial Binning of Color feature for my classifier.  Again I ran test with both having it on and off and found the classification worked better with it.   I experimented with various color spaces and image sizes.  Here is an example of using `YCrCb` color space, resized to 32x32:

![alt text][image7]
![alt text][image8]


#### 2. Explain how you settled on your final choice of HOG parameters.

I found that computing the HOG Descriptors took a long time with a small `pixels_per_cell` value.  I found that I could double the `pixels_per_cell` resuting in a third of the time to make a prediction.   This caused lower accuracy, but that could be mitigated by increasing the number of `orientations`, which had much less impact on pediction speed.

The final HOG hyperparameters was `orientations=12` and `pixels_per_cell=(16,16)`.

(**Note**: I acutally wrote a loop to test various combinations of hyper parameters, see section 3 below).

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]
![alt text][image2]

I performed a "parameter search" in code block 13 of the Jupyter notebook.  It looped over various combinations of SVM parameters (`kernel`, `C`) and HOG (`orientations`, `pixels_per_cell`).

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

I ended up using `orientations=12`, `pixels_per_cell=(16,16)`, `C=1`, `kernel=linear` as hyperparemters which had an Accuracy 99.24%, and a prection time of only 0.19 seconds.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

My approach to the sliding windows was to use smaller windows only at the horizon where cars would be distant and small, and use medium windows to search the mid region of the image, and large windows near the bottom where close (large) cars would be found.  

I found that more windows (even smaller ones near horizon and larger ones near bottom of the image) increased the chances of finding cars, but at the cost of greatly increasing the processing time for video, so I settled on the "small/medium/large" configuration.

Below is an image showing the sliding windows.

![alt text][image9]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Below is the classifier processing 6 example images:

![alt text][image10]


---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image11]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image12]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

