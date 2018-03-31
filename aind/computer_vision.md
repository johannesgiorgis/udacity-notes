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

