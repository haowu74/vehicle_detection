# Vehicle Detection
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_not_car.png
[image2]: ./output_images/HOG_visualization.jpg
[image3]: ./output_images/test5.jpg
[image4]: ./output_images/test5_final.jpg
[video1]: ./output_images/project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in function `get_hog_features()` in `functions.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters. From colour space's perspective, YCrCb seems the best option in my tests. On the other hand, HOG's performance varies a lot when using different channel: when channel = 0, 1, or 2,
the calculation was quick, while the result was terrible. Finally when I set the channel to ALL, it worked as miracle, although the scan speed is no longer suitable for any real-time application. It took a few hours 
to complete scan the project video, which is far from ideal using real-time application criteria. I haven't tried many variation of other HOG parameters, such as direction, pix_per_cell, and cell_per_block.
They does not seem as important as the channel number, so I just use the default ones demonstrated in the course's video.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using `LinearSVC()` in the function of `train_classifier()` in file `functions.py`. It uses the combination of HOG feature and YCrCb colour space feature. 

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?
The sliding window technique is used to scan the picture / frame of the video. The window uses 3 sizes: 64x64, 96x96 and 140x140. The range of scan are at the bottom half of the picture: from y = 340 downwards.
It seems quite good to set overlapping ratio to 0.8.
The algorithm is slow because for each frame, there are many clips to be used as the input of the SVM model. Each pixel is scanned multiple times.

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

After searching the pictures using the algorithms mentioned above, we use heat map to filter the squares. Here are some example images:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.
I use heatmap and threshold to sum up the overlapping squares being detected as vehicle. The codes are implemented in function of `add_heat()` and `apply_threshold()`. 
Then the final boxes are found by using `label()`.

---

### Discussion
This project took very long time to fine tune. I have tried a few combinations and found the HOG channel = ALL is a must, without which there is no chance to get a decent result. Colour feature seems 
not that important compared to HOG feature.
Before start working on this project, I have heard saying HOG+SVM is slow, but I did not expect that slow. It is impossible to use this algorithm in a real car. In addition,
this is for just training detecting car, then how about pedestrian, bicycle, moto-bike? Do we need to scan again and again for detect other important objects in traffic?
To be honest, this project shows us the traditional way to detect car. Yes now we know it is very slow, therefore very limited, in a hard way. Why not just learn Neural 
Network approaches such as Yolo, Faster R-CNN, or SSD in the first place?   
  

