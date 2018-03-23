# AIND - Computer Vision Notes
This contains notes and references from Udactiy AIND Term 2's Computer Vision section.

## Lesson 6 - Mimic Me!
Project: [AIND-CV-Mimic](https://github.com/udacity/AIND-CV-Mimic)



## Lesson 7 - Image Representation and Analysis
Computer Vision used in AI systems to visually perceive the world by gathering images, analyzing data, and eventually responding to it.


#### Computer Vision Pipeline
1. Input Data
	* Images or Images Frame  
2. Pre-Processing
	* Noice Reduction
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

Color to Grayscale:  
	1. Grayscale is more useful in recognizing objects.  
	2. Color images are harder to analyze and wake up more space in memory.  

#### Intensity
- measure of lightness and darkness in an image
- Patterns in lightness and darkness define the shape and characteristics of many objects
- intensity alone can provide enough information to identify objects and interpret an image correctly

Most simple identification tasks rely on identifying the shape and intensity patterns in objects, and grayscale images provide this information


#### When is Color Important?  
- In general, if objects or traits are easier to identify in color for us humans, it's better to provide color images to algorithms
- e.g. computer-aided diagnostics - color can be a good indicator of health, illnesss or other condition


#### Images as Functions
- treating images as functions is the basis for many image processing techniques
- e.g. geometrically warping the size and apparent shape of an image, changing appearance from color to grayscale
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
- Commonly used with bue screen
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
	- It is good practice to intially convert BGR images to RGB before analyzing and manipulating them


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
- A maching learning technique that seperates an image into segments by clustering/grouping together data points that have similar traits
- Unsupervised Learning method - does not rely on labeled data
- UL aims to find groupings and patterns among unlabeled datasets
- K-Means Clustering Algorithm:
	1. Choose k random center points
	2. Assign every data point to a cluster, based on its nearest center point
	3. Takes the mean of all the values in each cluster - these mean values become the new center points
	4. Repeats steps 2 and 3 until *convergence* is reached
- convergence is defined by us - either by number of iterations or number of times center points have moved



## Links
- [Affectiva's JS SDK documentation](https://affectiva.readme.io/docs/getting-started-with-the-emotion-sdk-for-javascript)
- [Image Coordinate Systems - Matlab documentation](https://www.mathworks.com/help/images/image-coordinate-systems.html)
- [OpenCV](http://opencv.org/)
- [OpenCV Website](http://opencv.org/about.html)
- [OpenCV cvtColor](http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_colorspaces/py_colorspaces.html)
- [OpenCV Documentation](http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_table_of_contents_contours/py_table_of_contents_contours.html)
- [OpenCV Contour Features](http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contour_properties/py_contour_properties.html)
- [OpenCV Probabilistic Hough Transform](http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html)
- [OpenCV K-Means](http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_ml/py_kmeans/py_kmeans_opencv/py_kmeans_opencv.html)
- [Latest CNN Segmentation Techniques](https://blog.athelas.com/a-brief-history-of-cnns-in-image-segmentation-from-r-cnn-to-mask-r-cnn-34ea83205de4)

