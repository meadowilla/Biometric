import cv2

def IrisEnhancement(img):
# perform the histogram equalization in each 32x32 region
    for row_index in range(0, img.shape[0], 32):
        for col_index in range(0, img.shape[1],32):
            sub_matrix = img[row_index:row_index+32, col_index:col_index+32]
            # apply histogram equalization in each 32x32 sub block
            img[row_index:row_index+32, col_index:col_index+32] = cv2.equalizeHist(sub_matrix.astype("uint8"))  
            
    return img