# Biometric

## Libraries
* numpy
* cv2
* datetime

## Following these steps in both training and testing
* Step 1: Read the image in grayscale mode (cv2.imread(image, 0))
* Step 2: Iris localization (finding pupil and iris)
* Step 3: Normalize the iris region
* Step 4: Enhance normalized iris image
* Step 5: Extract features from the enhanced image

* From step 1 to step 4: image preprocessing
* Step 5: feature extraction

## Iris Localization
1. Remove noise but also preseve edges (Why cv2.bilateralFilter(eye, d, sigmaColor, sigmaSpace)? which mathematic method?)
* d: Diameter of the pixel neighborhood used for filter. Larger value <=> more pixels around the target pixel used for filtering
* sigmaColor: Sigma values for the color space, controlling filter's sensitivity to color differences. Larger value <=> more colors considered similar, stronger smoothing.
* sigmaSpace: Sigma values for the coordinate space, controlling filter's sensitivity to spatial differences. Larger value <=> pixels farther from the target pixel will influence the filtering result more.

2. Initial estimate of the pupil's center 
* The pupil is the darkest region in the eye image. Identifying the column and row with the least sum, which likely intersect at the pupil's center.
* This estimate is then refined by examining a smaller region around this point.

3. Refinement Step
* The sub-region is a square centered at (Xp, Yp) with a fixed size (120 x 120 pixels).
* Sum pixel values in the sub-region and find the minimum sum indices to update the pupil center coordinates

4. Gaussian Blur for reducing noise and smooth the image. 
* (5, 5) is the size of Gaussian kernel.
* 0 is the standard deviation of the X direction.

5. Hough Circle Transform for detecting circles.
* blur: The input image (blurred sub-region).
* cv2.HOUGH_GRADIENT: The detection method.
* dp=1.2: The inverse ratio of the accumulator resolution to the image resolution.
* minDist=200: Minimum distance between the centers of detected circles.
* param1=200: Higher threshold for the Canny edge detector (used internally by the Hough Circle Transform).
* param2=12: Accumulator threshold for the circle centers at the detection stage Smaller values mean more false circles may be detected.
* minRadius=15: Minimum circle radius.
* maxRadius=80: Maximum circle radius.

6. Further Processing on the copy
* Adding 7 pixels to pupil radius - better isolating the pupil from the rest of the eye.
* Median blur 3 times, each with a kernel size of 11; for further smoothing and reducing noise.

7. Canny edge detection to find highlighted edges
* threshold1=15: The lower threshold for the hysteresis procedure
* threshold2=30: The upper threshold for the hysteresis procedure
* L2gradient=True: This flag indicates that the more accurate L2 norm should be used to calculate the gradient magnitude

8. Removing the edges of the pupil from the edge-detected image in order to isolate the iris edges in the subsequent steps.

9. Searching for the iris circles using Hough Circle Transform with different radii to find the most prominent one. If the detected iris circle is too far from the pupil center (Euclidean Distance), we adjust the iris center to match the pupil center.

## Iris Normalization
1. The dimensions of the normalized iris image: 64 rows and 512 columns.
2. Normalizes the iris region by mapping the pixels from the localized image to the normalized image.
3. Inverts the pixel values of the normalized image.

## Image Enhancement
* Using equalize of skimage.filters.rank and disk of skimage.morphology.
* Histogram equalization using a disk-shaped structuring element with a radius of 32 pixels (disk(32)) to improve contrast to improve the contrast of an image by redistributing the intensity values.
* Region of interest (ROI) consists of the first 48 rows and all columns of the normalized image.

## Feature Extraction
* Generating a Gabor kernel. what is Gabor kernel????
* Convolving the normalized image with Gabor




## Performance Evaluation
* table_CRR
* performance_evaluation

## Bootstrap
* A statistical method used to estimate the distribution of a statistic (mean or standard deviation) 
* By resampling with replacement from the original data
### Steps
* Resampling the data multiple times (specified by times).
* Calculating the False Match Rates (FMRs) and False Non-Match Rates (FNMRs) for each resampled dataset.
* Aggregating the results to compute mean and confidence intervals for the performance metrics.