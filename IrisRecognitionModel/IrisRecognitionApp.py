import os
import cv2
import numpy as np
from IrisEnhancement import IrisEnhancement
from IrisLocalization import IrisLocalization
from IrisNormalization import IrisNormalization
from FeatureExtraction import FeatureExtraction
from HammingDistance import find_min_hamming_distance
import tkinter as tk
from tkinter import filedialog, messagebox

# Define the threshold for iris verification
threshold = 0.34

# Define the function to calculate the iris code of an image
def calculate_iriscode(imgpath):
    eye = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
    iris, pupil = IrisLocalization(eye)
    normalized = IrisNormalization(eye, pupil, iris)
    enhanced = IrisEnhancement(normalized)
    feature = FeatureExtraction(enhanced, 16)
    return feature

# Define the function to register the user
def register_new_user():
    imgpath = filedialog.askopenfilename(title="Select Image for Registration")
    if imgpath:
        registered_iris_code = calculate_iriscode(imgpath)
        np.savetxt("registered_iris_code.txt", registered_iris_code, fmt='%d')
        messagebox.showinfo("Registration", "User registered successfully!")

# Define the function to verify the user
def verify_user():
    imgpath = filedialog.askopenfilename(title="Select Image for Verification")
    if imgpath:
        checkin_iris_code = calculate_iriscode(imgpath)
        registered_iris_code = np.loadtxt("registered_iris_code.txt", dtype=int)
        min_distance = find_min_hamming_distance(checkin_iris_code.tolist(), registered_iris_code.tolist())[0]
        print(f"min_distance = {min_distance}")
        if min_distance < threshold:
            messagebox.showinfo("Verification", "User verified successfully!")
        else:
            messagebox.showinfo("Verification", "User not verified!")

# Create the main window
root = tk.Tk()
root.title("Iris Recognition App")
root.geometry("500x300")
root.configure(bg="#f0f0f0")

# Create a frame for better layout
frame = tk.Frame(root, bg="#f0f0f0")
frame.pack(pady=30)

# Create a label for the title
title_label = tk.Label(frame, text="Iris Recognition App", font=("Helvetica", 20), bg="#f0f0f0", fg="#333333")
title_label.pack(pady=20)

# Create buttons for registration and verification
register_button = tk.Button(frame, text="Register User", command=register_new_user, width=25, height=2, bg="#4CAF50", fg="white", font=("Helvetica", 14))
register_button.pack(pady=15)

verify_button = tk.Button(frame, text="Verify User", command=verify_user, width=25, height=2, bg="#008CBA", fg="white", font=("Helvetica", 14))
verify_button.pack(pady=15)

# Run the main loop
root.mainloop()