

## Lesson 6 - Mimic Me!
Project: [AIND-CV-Mimic](https://github.com/udacity/AIND-CV-Mimic)

## Lesson 7 - Image Representation and Analysis
- computer vision used in AI systems to visually perceive the world by gathering images, analyzing data, and eventually responding to it.

Computer Vision Pipeline
	1. Input Data
		- Images or Images Frame
	2. Pre-Processing
		- Noice Reduction
		- Color Correction
		- Scaling
	3. Selecting Areas of Interest
		- Face Reduction
		- Image Cropping
	4. Feature Extraction
		- Finding Facial Markers (mouth, eyes...etc)
	5. Prediction/Recognition
		- Facial Expression Recognition
		- Emotion Prediction
	6. ACTION!


_Pre-Processing_
- all about making an image or sets of images easier to analyze and process computationally.
- 2 purposes:
	1. *Correct* images and eliminate unwanted traits.
	2. *Enhance* the most important parts of an image.

Color to Grayscale
	1. Grayscale is more useful in recognizing objects.
	2. Color images are harder to analyze and wake up more space in memory.

Intensity
- measure of lightness and darkness in an image
- Patterns in lightness and darkness define the shape and characteristics of many objects
- intensity alone can provide enough information to identify objects and interpret an image correctly

Most simple identification tasks rely on identifying the shape and intensity patterns in objects, and grayscale images provide this information

When is Color Important?
- In general, if objects or traits are easier to identify in color for us humans, it's better to provide color images to algorithms
- e.g. computer-aided diagnostics - color can be a good indicator of health, illnesss or other condition


Images as Functions
- treating images as functions is the basis for many image processing techniques
- e.g. geometrically warping the size and apparent shape of an image, changing appearance from color to grayscale
- Image processing transforms an image pixel by pixel


## Links
[Affectiva's JS SDK documentation](https://affectiva.readme.io/docs/getting-started-with-the-emotion-sdk-for-javascript)
[Image Coordinat Systems - Matlab documentation](https://www.mathworks.com/help/images/image-coordinate-systems.html)