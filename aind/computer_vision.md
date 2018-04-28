# AIND - Computer Vision Notes
This contains notes and references from Udacity AIND Term 2's Computer Vision section.

## Lesson 6 - Mimic Me!
Project: [AIND-CV-Mimic](https://github.com/udacity/AIND-CV-Mimic)



## Lesson 7 - Image Representation and Analysis
Computer Vision used in AI systems to visually perceive the world by gathering images, analyzing data, and eventually responding to it.


#### Computer Vision Pipeline
1. Input Data
	* Images or Images Frame  
2. Pre-Processing
	* Noise Reduction
	* Color Correction
	* Scaling  
3. Selecting Areas of Interest
	* Face Reduction
	* Image Cropping  
4. Feature Extraction  
	* Finding Facial Markers (mouth, eyes...etc)  
5. Prediction/Recognition
	* Facial Expression Recognition
	* Emotion Prediction
6. ACTION!  


#### Pre-Processing
- all about making an image or sets of images easier to analyze and process computationally.
- 2 purposes:
	1. *Correct* images and eliminate unwanted traits.
	2. *Enhance* the most important parts of an image.

Color to Gray-scale:  
	1. Gray-scale is more useful in recognizing objects.  
	2. Color images are harder to analyze and wake up more space in memory.  

#### Intensity
- measure of lightness and darkness in an image
- Patterns in lightness and darkness define the shape and characteristics of many objects
- intensity alone can provide enough information to identify objects and interpret an image correctly

Most simple identification tasks rely on identifying the shape and intensity patterns in objects, and gray-scale images provide this information


#### When is Color Important?  
- In general, if objects or traits are easier to identify in color for us humans, it's better to provide color images to algorithms
- e.g. computer-aided diagnostics - color can be a good indicator of health, illnesses or other condition


#### Images as Functions
- treating images as functions is the basis for many image processing techniques
- e.g. geometrically warping the size and apparent shape of an image, changing appearance from color to gray-scale
- Image processing transforms an image pixel by pixel
- Digital images are stored as matrices or 2D arrays. Each index in the matrix corresponds to one pixel in the displayed image
- image coordinate system: images are 2 dimensional and lie on the x-y plane, origin (0, 0) is at the top left of the image


**Quiz:**  
- Move the image 100 pixels up: `G(x,y) = F(x, y - 100)`  
- Darken the image: `G(x,y) = 0.5 * F(x, y)`  
- Create a purely black and white image (no gray): `G(x,y) = {0 if F(x,y) < 150, and 255 otherwise`  
- Move the image 100 pixels down: `G(x,y) = F(x, y+100)`  
- Subtract another image F2(x, y) of the same size from the original image: `G(x,y) = F(x,y) - F2(x,y)`


#### Color Thresholds
- Use information about the colors in an image to isolate a particular area  
- example of CV pipeline step: selecting areas of interest  
- select an area of interest using a color threshold  
- Color Threshold used in number of applications  
	- Computer graphics  
	- Video  
- Commonly used with blue screen
- Blue screen similar to green screen is used to layer two images or video streams based on identifying and replacing a large blue area.
- How does it work?
	- Isolate blue background
	- Replace blue area with an image of your choosing


#### Coding a Blue Screen
- OpenCV is a popular computer vision library that has many built in tools for image analysis and understanding
- **Why BGR instead of RGB?**
	- OpenCV reads in images in BGR format vs RGB as BGR color format was popular among camera manufacturers and image software providers when it was being developed
	- The red channel considered one of the least import color channels, so was listed last
	- The standard has changed and most image software and cameras use RGB format
	- It is good practice to initially convert BGR images to RGB before analyzing and manipulating them
- Good practice to always make a copy of the image to avoid modifying original image
- Masks are a common way to isolate a selected area of interest and do something with that area


#### Color Spaces and Transforms
- We saw how to detect a blue screen background
- This detection assumed
	- Scene was very well lit
	- Screen was a very consistent blue
- This would not work under varying light conditions
- So how can we consistently detect objects under varying light conditions?
	- there are many other ways to represent colors in an image besides red, blue, and green values.
	- these different color representations are often called 'color spaces'
- Color Spaces
	- 3D space where any color can be represented by a 3D coordinate of R, G, and B values.
- Other Color Spaces:
	- HSV: Hue, Saturation, Value
	- HLS: Hue, Lightness, Saturation
- HSV:
	- Isolates the Value (V) component, which varies the most under different lighting conditions
	- The Hue (H) channel stays fairly consistent in shadow or excessive brightness
- To select the most accurate color boundaries, it's often useful to use a color picker and choose the color boundaries that define the region you want to select
- HSV space is more valuable in selecting an area under varying light conditions


#### Geometric Transformations
- Move pixels around in an image based on a mathematical formula
- Useful for changing an image to a desired perspective - e.g. taking an angled image of a text and transforming it to a straight version
- Transforms the Z-coordinate of the object points, which changes that object's 2D image representation
- It warps the image and effectively drags points towards or pushes them away from the camera to change the apparent perspective
- It can also transform the X and Y image pixels so that the entire image is rotated or moved slightly from the original perspective
- Transforming the perspective of an image allows you to more reliably measure or recognize certain image traits
	- measure the curvature of a road lane from a bird's eye view of a road
	- more easily read important text that's written at an angle
- Common use is in scanning and aligning text in a document - e.g. banks and reading/scanning checks, apps that scan/read everything
- Computer vision is used to align document scans and improve readability
- Business Card Reader
	- Take a picture of a business card
	- Straighten and align the image
	- Use text recognition to automatically read in contact info


#### Filters Revisited
- In addition to taking advantage of color information and moving pixels around, we also have knowledge about patterns of intensity in an image. We can use this knowledge to detect other areas of visual traits of interest
- Edges occur when an image changes from a very dark to light area
- These edges often define object boundaries which help us distinguish and eventually identity these objects
- Edge detection filters also known as High-pass Filters. They detect big changes in intensity or color in an image and produce an output that shows these edges
- Low-pass Filters used to pre-process an image by reducing noise or unwanted traits in an image


#### Frequency in Images
- Frequency in images is a rate of change
- High frequency image is one where the intensity changes a lot
- Low frequency image may be one that is relatively uniform in  brightness or changes very slowly
- Most images have both high-frequency and low-frequency components.


#### High-pass Filters
- Filters used to filter out unwanted or irrelevant information in an image OR to amplify features like object boundaries or other distinguishing traits.
- Sharpen an image
- Enhance _high-frequency_ parts of an image
	- areas where the levels of intensity in neighboring pixels rapidly change like from very dark to very light pixels
- Since we're looking at patterns of intensity, the filters will operate on gray-scale images that represent this information and display patterns of lightness and darkness in a simple format
- A high-pass filter will block out areas where there is no or little change in intensity and turn these pixels black. In areas where a pixel is way brighter than its immediate neighbors, it will enhance that change and create a line. These emphasizes edges.
- Edges are areas in an image where the intensity changes very quickly and they often indicate object boundaries


**Convolution Kernels**  
- a kernel is a matrix of numbers that modifies an image  
- for edge detection, it is important all the edges add up to 0, because this filter is computing the difference or change between neighboring pixels. Differences are calculated by subtracting pixel values from one another  
- If these kernel values don't add up to 0, the calculated difference will be positively or negatively weighted, which will have the effect of brightening or darkening the entire filtered image respectively  
- Kernel convolution is an important operation in Computer Vision Applications. It is the basis for convolutional neural networks.  
	- Involves taking a kernel, a small grid of numbers and passing it over an image pixel by pixel transforming it based on what these numbers are  
	- By changing these numbers, we can create many different effects from edge detection to blurring an image  
- Kernel convolution relies on centering a pixel and looking at it's surrounding neighbors. So how do you handle image corner or edges?  
	- Extend _(default)_: the nearest border pixels are conceptually extended as far as necessary to provide values for the convolution  
	- Wrap: The image is conceptually wrapped (or tiled) and values are taken from the opposite edge or corner  
	- Crop: Any pixel in the output image which would require values from beyond the edge is skipped. Output image can be slightly smaller  

**Quiz**  
- How best to find and enhance horizontal edges and lines in an image?  
	- A kernel that finds the difference between the top and bottom edges surrounding a given pixel  


#### Gradients and Sobel Filters
- Gradients are a measure of intensity change in an image; they generally mark object boundaries and changing area of light and dark
- Sobel filter is very commonly used in edge detection and in finding patterns in intensity in an image
	- Applying a Sobel filter to an image is a way of **taking (an approximation) of the derivative of the image** in the _x_ or _y_ direction
	- Taking the gradient in the _x_ direction emphasizes edges closer to vertical
	- Taking the gradient in the _y_ direction emphasizes edges closer to horizontal
	- It also detects which edges are _strongest_. This is encapsulated by the **magnitude** of the gradient. A stronger edge has a greater magnitude


#### Low-pass Filters
- Noise in an image appears as speckle or discoloration in an image
- Noise might mess with processing steps:
	- Edge detection when high pass filters can amplify noise if it's not removed first
- Most common way to remove noise is via a low-pass filter
- Low-pass filter can
	- Blur/smooth an image
	- Block high-frequency parts of an image
- Very useful in medical images which typically have noise that is reduced by the imagery machinery or by a moving human subject
- Low Pass Filters
	- Averaging Filter: 3 x 3 kernel with weights that send the same amount to a pixel and its surrounding pixels
	- Typically take an average and not a difference as high pass filters do, so their components should all add up to one
	- This preserves the image brightness and that it doesn't get brighter or darker overall
	- If surrounding pixels are mostly brighter than the center pixel, the new output pixel value is brighter as well
	- we get an average to smoothed out image with fewer abrupt changes in intensity
	- this is useful for blowing out noise or making a background area within a certain intensity range look more uniform
	- Used in Photoshop to soften and blur parts of an image


#### Gaussian Blur
- Blur/smooth an image
- Blocks high-frequency parts of an image
- Preserves edges! (difference from Low-pass filters)
- Perhaps the most frequently used low-pass filter in computer vision applications
- a weighted average that gives the most weight to the center pixel while still taking into account the surrounding pixels more so depending on how close they are to the center


#### Canny Edge Detector
- Edge Detection
	1. gray-scale
	2. low-pass filter
	3. high-pass filter
	4. binary threshold
- Edge detection still a complex problem even with all these tools used together. We have to think about:
	- What level of intensity change constitutes an edge?
	- How can we consistently represent thick and thin edges?
- Canny Edge Detector
	- One of the best and most frequently used edge detectors that takes all of these questions into account is the canny edge detector
	- Widely used and accurate edge detection algorithm
	- Goes through a series of steps that consistently produce accurately detected edges
		1. **Filters our noise** using a Gaussian blur
		2. **Finds the strength and direction of edges** using Sobel filters
		3. **Applies non-maximum suppression** to isolate the strongest edges and thin them to one-pixel wide lines
		4. Uses **hysteresis to isolate the best edges**
- Hysteresis is a double thresholding process
	- we define a high threshold that allows strong edges to pass through
	- we define a low threshold to discard any edge below this threshold
	- an edge whose strength falls in between this low threshold and high threshold will be kept only when it's connected to another strong edge
	- Canny eliminates weak edges and noise; isolates edges that are most connected and are therefore most likely to be part of an object boundary


#### Review
- treating images as functions so we can perform operations on pixel values and locations to transform an image
- we transformed the color and geometry of images and filtered our images to enhance the _most important_ information
- we learned how to implement one BIG step of the computer vision pipeline: **Pre-processing**



## Lesson 8 - Image Segmentation
#### Image Segmentation:
- The process of dividing an image into segments or unique areas of interest
- Done in 2 main ways:
	* By connecting a series of detected edges
	* By grouping an image into separate regions by area or distinct traits


#### Image Contours
- Edge detection algorithms used to detect the boundaries of objects
- This also highlights interesting features and lines. But to do image segmentation, you only want complete closed boundaries - e.g. the outline of a hand, not necessarily the interesting features within the boundary of the hand.
- Image Contours:
	* continuous curves that follow the edges along a perceived boundary
	* provide a lot of information about the shape of an object boundary
	* Detected when there is a white object against a black background
	* So we have to create a binary threshold of an image, which has black and white pixels that distinguish different objects in an an image
	* Then we use edges of these objects to form contours

- Extract lots of information about the shape of the hand called Contour Features
	* Area
	* Center
	* Perimeter
	* Bounding Rectangle


#### Hough Transform
- Simplest boundary you can detect is a line
- More complex boundaries are often made up of several lines
- Represent any line as a function of space
- Hough transformation converts a line in image space to a point in Hough space


#### K-Means Clustering
- Commonly used image segmentation technique
- A machine learning technique that separates an image into segments by clustering/grouping together data points that have similar traits
- Unsupervised Learning method - does not rely on labeled data
- UL aims to find groupings and patterns among unlabeled datasets
- K-Means Clustering Algorithm:
	1. Choose k random center points
	2. Assign every data point to a cluster, based on its nearest center point
	3. Takes the mean of all the values in each cluster - these mean values become the new center points
	4. Repeats steps 2 and 3 until *convergence* is reached
- convergence is defined by us - either by number of iterations or number of times center points have moved


#### Review
- Looked selecting areas by using information about color, geometry, and patterns of intensity in images
- Went through many **image segmentation** techniques that move us closer to the end of the pipeline: object recognition, and scene understanding



## Lesson 9 - Features and Object Recognition
#### Features and Object Recognition
- We have learned about the first half of the computer vision pipeline:
	- Input Data
	- Preprocessing (Image Processing)
	- Selecting areas of interest (Image Segmentation)
- The remaining steps are:
	- Feature extraction
	- Image Recognition
- Features are distinct and measurable pieces of information in an image. They are the basis of many machine learning and pattern recognition techniques


#### Why Use Features?
- A feature is a measurable piece of data in an image
	- distinct color in an image
	- specific structure such as a line, an edge or an image segment
- A good feature will help us recognize an object in all the ways it may appear. It is easily tracked and compared. It should be consistent across different scales, lighting conditions and viewing angles. It will also be ideally visible in noisy images and in images where only part of an object is visible
- e.g. recognizing a bike from the side, front, far away or closer to you
- Typically, features are quite small and sets of them are used to identify larger objects
- Most important quality is repeatability, which is whether or not the feature will be detected in 2 or more different images of the same object or scene
- Feature extraction is used to reduce the dimensionality of image data
- By isolating specific color or spatial information, feature extraction can transform complex and large images into smaller sets of features
- The task of classifying images based on just their features becomes simpler and faster


#### Types of features
- 3 categories of features
	- edges
	- corners
	- blobs
- Edges: areas in an image where the intensity abruptly changes. Also known as areas that have a high intensity gradient
- Corners: found at the intersection of 2 edges, form what looks like a corner or sharp point
- Blobs: Region-based features that may include areas of extreme highs or lows in intensity or areas of a unique texture
- We are most interested in detecting corners. They are the most repeatable features, they are easy to recognize given two or more images of the same scene
- A corner represents a point where two edges change. If we move either of those up or down, the corner patch will not match exactly with that area. Corners are easiest to match and make good features because they are so unique


#### Corner Detectors
- When building an edge detector, we looked at the difference in intensity between neighboring pixels. An edge was detected if there was a big and abrupt change in intensity in any one direction - up or down, left or right, or diagonal.
- Change in intensity in an image is also referred to as the image gradient
- We can also detect corners by relying on these gradient measurements
- Corners are the intersection of two edges. We can detect them by taking a window which is generally a square area that contains a group of pixels and looking at where the gradient is high in all directions.
- Each of these gradient measurements has an associated magnitude which is a _measurement of the strength of the gradient_, and a direction which is the _direction of the change in intensity_
- Both of these values can be calculated by Sobel operators
- Sobel operators take the intensity change/gradient of an image in the x and y direction separately
- To get magnitude and direction of the total gradient from the x and y gradients, we convert from image space to polar coordinates
- Mini corner detectors:
	1. Shift a window around an area in an image
	2. Check for a **big variation** in the direction and magnitude of the calculated gradients. This large variation identifies a corner	
- Dilation enlarges bright regions or regions in the foreground so we can see them better
- Corners alone can be useful for many types of analysis and geometric transformations


#### Dilation and Erosion
- Dilation and erosion known as **morphological operations** - often performed on binary images, similar to contour detection
- Dilation enlarges bright, white areas in an image by adding pixels to the perceived boundaries of objects in that image
- Erosion removes pixels along object boundaries and shrinks the size of objects
- Both operations are performed in sequence often to enhance important object traits (similar to how low and high pass filters are combined together)
- Dilate
	- via _dilate_ function
	- 3 inputs: an ordinary binary image, kernel that determines the size of the dilation (None results in default size), number of iterations to perform the dilation (typically 1)
- Erode
	- via _erode_ function
	- 3 inputs: an ordinary binary image, kernel that determines the size of the erosion (None results in default size), number of iterations to perform the erosion (typically 1)
- One combination of erosion and dilation operations is **opening**, which is **erosion followed by dilation**. Useful in noise reduction
	- use _morphologyEx_ function
	- `opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)`
- Another combination is **closing** - **dilation followed by erosion**. Useful in _closing_ small holes or dark areas within an object
- Many of these operations try to extract better or less noisy information about the shape of an object or enlarge important features


#### Feature Vectors
- Object detection relies on recognizing distinct sets of features, so we have to look at distinct sets of features often called **feature vectors**
- If you look at the direction of multiple gradients around the center point of an image, you can get a feature vector that makes for a robust representation of the shape of an object
- E.g. image of trapezoid. Break up its edge detected image into a grid of cells and look at the direction of the gradient of each cell with respect to the trapezoid, we can flatten this data and create a 1D array. This is a feature vector, specifically a vector of gradient directions
- Same concept can be applied to a circle as well
- These 2 feature vectors exactly represent these two shapes (trapezoid and circle)
- Ideally, to accurately identify any circle or trapezoid at different sizes or from different perspectives, these vectors should allow for enough flexibility to detect some variation in these shapes while remaining distinct enough to be able to distinguish different shapes


#### HOG
- Many algorithms designed to extract spatial features and identify objects using information about image gradients
- 1 illustrative technique is HOG - Histogram of Oriented Gradients
- Histogram is a graphical representation of the distribution of data
- Oriented Gradients is the direction of image gradients
- HOG should produce a histogram of gradient directions in an image
	1. Calculates the magnitude and direction of the gradient at each pixel
	2. Groups these pixels into square cells (typically 8x8)
	3. Counts how many gradients in each cell fall in a certain range of orientation and sums the magnitude of these gradients so that the strength of the gradients are accounted for
	4. Places all that directional data into a histogram
- HOG is actually a feature vector
- Next step is to use these HOG features to train a classifier. Among images of the same object at different scales and orientations, this same pattern of HOG features can be used to detect the object wherever and however it appears


#### Implementing HOG
- HOG also referred to as a type of **feature descriptor** - a simplified representation of an image that is made up of extracted features (that highlight the important parts of an image) and that discards extraneous information
- Number of steps involved to create a HOG feature vector
- Many image sets _require_ pre-processing as a first step to ensure consistency in size and color
- Implementing HOG
	1. Calculate image gradient at each pixel (magnitude & direction) via Sobel filters (use OpenCV's Sobel function to avoid creating your own)
	2. Define how we divide this data into a histogram. Use 9 bins for 9 different ranges of gradient directions
	3. Calculate the histogram for each cell. We have 64 total 8x8 cells, for each of these we calculate a histogram of directions (9 bins) weighted with their magnitude. So each cell will produce a feature vector containing 9 values. 64 of those together give us a complete image feature vector containing 576 values. This is _almost_ the feature vector we use to train our data.
	4. One more step HOG does before creating the feature vector is to perform _block normalization_. A block is a larger area than a cell and checks different positions for the cell, determining how much they overlap. So HOG features for all cells in each block are computed at each block position and the block shifts across and down through the image cell by cell.
- The actual number of features in your final feature vector will be the total number of block positions multiplied by the number of cells per block times the number of orientations: `7 x 7 x 2 x 2 x 9 = 1764` [Where is the 7 coming from?]
- Example
	- Pixels per cell: `8 x 8`
	- Cells per block: `2 x 2`
	- Number of orientations: `9`
- OpenCV has a function to help us perform the complete HOG algorithm with defined blocks and cells - `cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, n_bins)`
- The feature vector that this produces is what you can use to **train a classifier**!
- QUIZ: Histograms are often used in forming useful features because of how they divide data into ranges using bins
	- Bins reduce the dimensionality of data; they represent image data in a more compact way
	- Grouping data into ranges allows for more flexibility in identifying similar objects based on their feature vectors


#### Object Recognition
- Now that we learned how to detect features from a variety of images, we can use them to classify objects
- Let's start by classifying a simple object - a banana
- We can build a simple classifier and train it to learn the difference between two types of image data - positive and negative data.
- Positive data are areas that include a banana
- Negative data area areas that don't include a banana
- We need labeled data that contains examples of both scenarios


#### Train a Classifier
- There are a number of ways to create a classifier
- In Machine Learning, we train a classifier by using Supervised Learning to recognize certain sets of features.
- Supervised Learning uses labeled data to check if it's classifying data correctly
- We need a big training set with images of the object we want to identify and images without it. We needs lots of positive and negative data
- Procedure
	- For each image in this training set, we will extract features and give them to a training algorithm along with their corresponding labels
	- Training algorithm will initialize a model and tweak its parameters as it tries to correctly match feature vectors with their correct labels
	- Iterative process where the model is given all this training data and predicts, based on a given feature vector, what its label will be.
	- Then it compares this with the actual true label. If the algorithm is wrong and has some error in its prediction, the model will adjust its parameters so that it correctly labels that feature vector.
	- It does this again and again until the error falls below a threshold or enough iterations have been passed
	- Model is said to be sufficiently trained
	- Next step is to see how this model performs on unseen data using test set of data, which we know the labels but our model doesn't.
	- We look at the test error to see how accurate our model truly is
- There are many great deep learning models such as convolutional neural networks that excel at recognizing patterns and features and detecting many types of objects
- There are also faster models that use some of the same kind of machine learning data separation techniques we've seen before, including:
	- Support Vector Machines (work well with HOG features)
	- Decision Trees
- There are other models you could use as well as combinations of models that you might choose depending on how fast or complex you may want a model to be.


#### Support Vector Machine Classifier
- SVM's work to best separate labeled data into groups and train until they reach an acceptably low error rate
- In the case of images and HOG feature classification, the SVM will train on sets of labeled images that also have associated HOG feature vectors. It will learn the association and try to classify new images by looking at their HOG feature vector.
- Using feature vectors is a lot faster than looking at large image files in their entirety	
- SVM's in OpenCV
	- To create an SVM in OpenCV, we define it's parameters and call a constructor
	- Next, prepare training data, associating images with their labels and computed HOG feature vectors. You should have as many sets of feature vectors for as many labels as you want to detect
	- Finally, you'll need to test your model to verify it's classification accuracy


#### Haar Cascades
- Haar Cascades, an algorithm that trains on many positive images and negative images
	- It detects Haar features by applying a Haar feature detector (e.g. vertical line detector)
	- It then performs classification on the entire image
	- If it doesn't get enough of a feature detection response, it classifies an area of an image as 'not face' and discards it
	- It feeds this reduced image area to the next feature detector (e.g. rectangle feature detection) and classified the image again, discarding irrelevant non-face areas at every step
	- This is called a **cascade of classifiers**
- Haar features are gradient measurements that look at rectangular regions around a certain pixel area and somewhat subtract these areas to calculate a pixel difference. Similar to how convolutional kernels work.
- Haar features detect patterns like edges, lines and more complex rectangular patterns
- In face detection, lines and rectangles are especially useful features because patterns of alternating bright and dark areas define a lot of features on a face
- Haar features effectively identify extreme areas of bright and dark on a face
- Haar cascades focus on processing and recognizing only the area in an image that's been classified as part of a face already and quickly throwing out irrelevant image data makes this algorithm very fast
- It is fast enough for processing a video stream in real time on a laptop computer
- Haar cascades may also be used for selecting an area of interest for further processing


#### Face Detection with OpenCV
- OpenCV comes with a few Haar Cascade detectors already trained
- code below detects faces
```python
import numpy as np
import cv2

## Load in the face detector XML file
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Read in an image
image = cv2.imread('face1.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect the faces in the image
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
```


#### Motion
- Same processing techniques can be applied to video streams as static images
- A video stream is made up of a sequence of image frames
- One unique aspect of these sequences of image frames is the idea of motion
- To create an intelligent computer vision system, we also want to give computers a way to understand motion
- This useful in a number of applications
	- isolate moving pedestrians from a still background
	- intelligent navigation systems
	- movement prediction models
	- distinguishing behaviors like running vs walking in a given video
- One way to track objects over time and detect motion is by extracting certain features and observing how they change from one frame to the next


#### Optical Flow
- Used in many tracking and motion analysis applications
- It works by assuming 2 things about image frames
	1. Pixel intensities of an object stay consistent between frames
	2. Neighboring pixels have similar motion
- How it works:
	- It looks at interesting points (corners or particularly bright pixels) and tracks them from one frame to the next
	- Tracking a point provides information about the **speed** of movement and data that can be used to **predict the future location** of the point
- Use Optical flow for tracking applications:
	- Hand gesture recognition
	- Tracking vehicle movement
	- Running vs. Walking
	- Safety applications by predicting the motion of things and performing obstacle avoidance like in the case of self-driving cars
	- Tracking eye movement for virtual reality games and advertising


#### Object Tracking
- Optical flow uses object features and and an area of perceived movement to track points and objects in consecutive frames
- To perform optical flow on a video or series of images, we first identify a set of feature points to track via a Harris Corner Detector or other feature detector. Next, at each time step or video frame, we track those points using optical flow
- OpenCV provides the function `calcOpticalFlowPyrLK()` for this purpose; it takes in 3 parameters
	- previous frame
	- previous feature points
	- next frame
- Using only this knowledge, it returns the predicted _next_ points in the future frame
- This way, we can track any moving object and determine how fast it's going and where it's likely to move next!


#### Outro
- Solid understanding of the computer vision pipeline from start to finish
- Starting from image processing techniques, built on that foundational knowledge to see how to recognize objects and interpret images
- With these skills, you'll be able to build computer vision applications of your own
- Object recognition and scene understanding are still active areas of research
- There are so many things to create in the world of robotics, medicine, assistive technology, virtual reality, and emotionally intelligent systems


## Project: CV Capstone Project
- [Project Github Repository](https://github.com/udacity/AIND-CV-FacialKeypoints)
- [OpenCV History Wikipedia](https://en.wikipedia.org/wiki/OpenCV#History)
- [Haar feature-based cascade classifiers](http://docs.opencv.org/trunk/d7/d8b/tutorial_py_face_detection.html)
- [Haar Cascades Pre-trained Architectures Github Repository](https://github.com/opencv/opencv/tree/master/data/haarcascades)
- [OpenCV Colorspaces](http://docs.opencv.org/3.2.0/df/d9d/tutorial_py_colorspaces.html)
- [Stackoverflow Recommended values for OpenCV DetectMultiScale Parameters](https://stackoverflow.com/questions/20801015/recommended-values-for-opencv-detectmultiscale-parameters)
- [Image Noise](https://digital-photography-school.com/how-to-avoid-and-reduce-noise-in-your-images/)
- [Image Denoising Example](http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_photo/py_non_local_means/py_non_local_means.html)
- [Gaussian blur Wikipedia](https://en.wikipedia.org/wiki/Gaussian_blur#Common_uses)
- [OpenCV Canny Detector](http://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/canny_detector/canny_detector.html)
- [OpenCV Smoothing Images](https://docs.opencv.org/3.1.0/d4/d13/tutorial_py_filtering.html)
- [Google Street View Maps](https://www.google.com/streetview/)
- [Kaggle Facial Keypoints Detection Data](https://www.kaggle.com/c/facial-keypoints-detection/data)
- [Using CNNs to Detect Facial Keypoints Tutorial](http://danielnouri.org/notes/2014/12/17/using-convolutional-neural-nets-to-detect-facial-keypoints-tutorial/)

**Keras**  
- [Sequential Model Methods](https://keras.io/models/sequential/#sequential-model-methods)  
- [Optimizers](https://keras.io/optimizers/)  
- [How can I save a Keras Model](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model)  
- [Display Deep Learning Model Training History in Keras](https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/)  


## Links
- [Affectiva's JS SDK documentation](https://affectiva.readme.io/docs/getting-started-with-the-emotion-sdk-for-javascript)
- [Image Coordinate Systems - Matlab documentation](https://www.mathworks.com/help/images/image-coordinate-systems.html)
- [OpenCV](http://opencv.org/)
- [OpenCV Website](http://opencv.org/about.html)
- [OpenCV cvtColor](http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_colorspaces/py_colorspaces.html)
- [Color Picker](https://www.w3schools.com/colors/colors_picker.asp)
- [Transformation Matrix](https://en.wikipedia.org/wiki/Transformation_matrix)
- [OpenCV Geometric Transformations](http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html)
- [OpenCV filter2D](http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_filtering/py_filtering.html)
- [Gaussian Blur](https://en.wikipedia.org/wiki/Gaussian_blur)
- [OpenCV GaussianBlur](http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_filtering/py_filtering.html#gaussian-filtering)
- [OpenCV Canny](http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_canny/py_canny.html)
- [OpenCV Documentation](http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_table_of_contents_contours/py_table_of_contents_contours.html)
- [OpenCV Contour Features](http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contour_properties/py_contour_properties.html)
- [OpenCV Probabilistic Hough Transform](http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html)
- [OpenCV K-Means](http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_ml/py_kmeans/py_kmeans_opencv/py_kmeans_opencv.html)
- [Latest CNN Segmentation Techniques](https://blog.athelas.com/a-brief-history-of-cnns-in-image-segmentation-from-r-cnn-to-mask-r-cnn-34ea83205de4)

- [OpenCV Corner Detection](http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_features_harris/py_features_harris.html)
- [Histograms in OpenCV](http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_histograms/py_table_of_contents_histograms/py_table_of_contents_histograms.html)
- [OpenCV Sobel function](http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_gradients/py_gradients.html)
- [Support Vector Machine Wiki](https://en.wikipedia.org/wiki/Support_vector_machine)
- [Kaggle](https://www.kaggle.com/)
- [ImageNet](http://www.image-net.org/)
- [OpenCV Haar Cascades](http://docs.opencv.org/3.0-beta/doc/user_guide/ug_traincascade.html?highlight=train%20cascade)
- [OpenCV Object Tracking](http://docs.opencv.org/trunk/d7/d8b/tutorial_py_lucas_kanade.html)
