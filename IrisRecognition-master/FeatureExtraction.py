import cv2
import numpy as np

def apply_gabor_filter(image, theta):
    # Create a Gabor filter
    sigma = 4.0
    wavelength = 8.0
    kernel_size = (31, 31)
    kernel = cv2.getGaborKernel(kernel_size, sigma, theta, wavelength, 0.5, 0, ktype=cv2.CV_32F)
    filtered = cv2.filter2D(image, cv2.CV_32F, kernel)
    return filtered

def FeatureExtraction(normalized_image):
    # Apply Gabor filters at different orientations
    orientations = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    iris_code = []

    for theta in orientations:
        response = apply_gabor_filter(normalized_image, theta)
        
        # Quantize responses
        real_response = np.real(response)
        imaginary_response = np.imag(response)
        
        # Convert to binary
        real_binary = (real_response > 0).astype(int)
        imaginary_binary = (imaginary_response > 0).astype(int)
        
        # Append to iris code
        iris_code.append(real_binary)
        iris_code.append(imaginary_binary)
    
    # Combine binary codes into a single array
    iris_code = np.concatenate(iris_code, axis=1) 
    return iris_code

