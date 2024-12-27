import cv2
import numpy as np

def gabor_filter_bank(scales, orientations, ksize=31, sigma=2.24):
    """
    Generates a bank of Gabor filters.
    :param scales: Number of scales.
    :param orientations: Number of orientations.
    :param ksize: Size of the Gabor filter kernel.
    :param sigma: Standard deviation of the Gaussian envelope.
    :return: List of Gabor filters.
    """
    filters = []
    for scale in range(scales):
        for orientation in range(orientations):
            theta = orientation * np.pi / orientations
            lambd = 4
            gamma = 1
            kernel_real = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi=0, ktype=cv2.CV_64F)
            kernel_imag = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi=np.pi/2, ktype=cv2.CV_64F)
            filters.append((kernel_real, kernel_imag))
    
    # Access the first Gabor filter (real and imaginary parts)
    kernel_real = filters[0][0]
    kernel_imag = filters[0][1]

    # Visualize the first Gabor filter (real and imaginary parts)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(kernel_real, cmap='gray')
    plt.title('Real Part of First Gabor Filter')
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.imshow(kernel_imag, cmap='gray')
    plt.title('Imaginary Part of First Gabor Filter')
    plt.colorbar()

    plt.show()
    return filters

def apply_gabor_filters(image, filters):
    """
    Apply a bank of Gabor filters to an image.
    :param image: Input normalized iris image.
    :param filters: List of Gabor filters.
    :return: List of filtered responses (real and imaginary).
    """
    responses = []
    for kernel_real, kernel_imag in filters:
        # Apply the real part of the filter
        response_real = cv2.filter2D(image, cv2.CV_64F, kernel_real)
        # Apply the imaginary part of the filter
        response_imag = cv2.filter2D(image, cv2.CV_64F, kernel_imag)
        # Compute the magnitude of the response
        magnitude = np.sqrt(response_real**2 + response_imag**2)
        responses.append((response_real, response_imag, magnitude))
    return responses

def phase_quantization(responses):
    """
    Quantize the phase of Gabor filter responses into 2-bit binary codes.
    :param responses: List of Gabor filter responses (complex values).
    :return: Binary feature vector (iris code).
    """
    binary_code = []
    for response in responses:
        real = response[0]
        imag = response[1]
        phase = np.arctan2(imag, real)
        quantized_phase = ((phase >= 0) & (phase < np.pi/2)) * 0b00 + \
                          ((phase >= np.pi/2) & (phase < np.pi)) * 0b01 + \
                          ((phase >= -np.pi) & (phase < -np.pi/2)) * 0b10 + \
                          ((phase >= -np.pi/2) & (phase < 0)) * 0b11
        
        # Convert each quantized value to 2 bits (binary representation)
        binary_block = np.unpackbits(np.uint8(quantized_phase).reshape(-1, 1), axis=1)[:, -2:]
        binary_code.append(binary_block.flatten())
  
    return np.hstack(binary_code)

def FeatureExtraction(image, block_size, scales=1, orientations=4):
    """
    Generate the binary feature vector (iris code) for an input image.
    :param image: Normalized iris image.
    :param block_size: Size of the blocks for pixel averaging.
    :param scales: Number of scales for Gabor filters.
    :param orientations: Number of orientations for Gabor filters.
    :return: Binary feature vector (iris code).
    """
    # Step 1: Divide the image into blocks and calculate block mean
    h, w = image.shape
    h_blocks, w_blocks = h // block_size, w // block_size
    block_means = np.zeros((h_blocks, w_blocks))
    for i in range(h_blocks):
        for j in range(w_blocks):
            block = image[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size]
            block_means[i, j] = np.mean(block)

    # Step 2: Generate Gabor filter bank
    filters = gabor_filter_bank(scales=scales, orientations=orientations)

    # Step 3: Apply Gabor filters to the block means
    responses = apply_gabor_filters(block_means, filters)

    # Step 4: Quantize phase to generate binary iris code
    iris_code = phase_quantization(responses)

    return iris_code
