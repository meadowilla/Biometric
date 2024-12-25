import os
import random
import numpy as np
from HammingDistance import HammingDistance 
import PerformanceEvaluation as PE

# Function to load dataset structure
def load_dataset_structure(root_path):
    dataset = {}
    for person_id in os.listdir(root_path):
        person_path = os.path.join(root_path, person_id)
        if os.path.isdir(person_path):
            dataset[person_id] = {}
            for session_id in os.listdir(person_path):
                session_path = os.path.join(person_path, session_id)
                if os.path.isdir(session_path):
                    dataset[person_id][session_id] = [
                        os.path.join(session_path, img) for img in os.listdir(session_path)
                    ]
    return dataset

# Function to generate pairs and compute distances
def create_pairs_and_labels(dataset, num_pairs=100):
    hamming_distances = []
    labels = []

    # Get all person IDs
    person_ids = list(dataset.keys())

    # Generate genuine pairs
    for _ in range(num_pairs // 2):
        person_id = random.choice(person_ids)
        sessions = dataset[person_id]
        all_images = [img for session in sessions.values() for img in session]
        if len(all_images) < 2:
            continue  # Skip if not enough images for a genuine pair
        img1, img2 = random.sample(all_images, 2)
        code1 = img1.replace("CASIA Iris Image Database (version 1.0)", "Iris Codes")
        code1 = code1.replace(".bmp", ".txt")
        code2 = img2.replace("CASIA Iris Image Database (version 1.0)", "Iris Codes")
        code2 = code2.replace(".bmp", ".txt")
        code1 = np.loadtxt(code1, dtype=int)
        code2 = np.loadtxt(code2, dtype=int)


        hamming_distances.append(HammingDistance(code1, code2))
        labels.append(1)  # Genuine pair

    # Generate impostor pairs
    for _ in range(num_pairs // 2):
        person1, person2 = random.sample(person_ids, 2)
        img1 = random.choice(
            [img for session in dataset[person1].values() for img in session]
        )
        img2 = random.choice(
            [img for session in dataset[person2].values() for img in session]
        )

        # Extract iris codes from images
        code1 = img1.replace("CASIA Iris Image Database (version 1.0)", "Iris Codes")
        code1 = code1.replace(".bmp", ".txt")
        code2 = img2.replace("CASIA Iris Image Database (version 1.0)", "Iris Codes")
        code2 = code2.replace(".bmp", ".txt")
        code1 = np.loadtxt(code1, dtype=int)
        code2 = np.loadtxt(code2, dtype=int)

        hamming_distances.append(HammingDistance(code1, code2))
        labels.append(0)  # Impostor pair

    return np.array(hamming_distances), np.array(labels)

# Example usage
root_path = "D:\\at school\\2024.1\\Biometric Authentication System\\Project\\Biometric\\IrisRecognitionModel\\CASIA Iris Image Database (version 1.0)\\"  # Replace with the actual path to your dataset
dataset = load_dataset_structure(root_path)
hamming_distances, labels = create_pairs_and_labels(dataset, num_pairs=100)

thresholds = list(np.arange(0.335, 0.375, 0.001))
fm1s = []
fnm1s = []
for threshold in thresholds:
    print("Threshold: ", threshold)
    fm1, fnm1, tpr1, fpr1 = PE.calculate_metrics(hamming_distances, labels, threshold)
    print("FMR: ", fm1)
    print("FNMR: ", fnm1)
    print("TPR: ", tpr1)
    print("FPR: ", fpr1)
    fm1s.append(fm1)
    fnm1s.append(fnm1)

eer, eer_threshold = PE.calculate_eer(np.array(fm1s), np.array(fnm1s), thresholds)
print("EER: ", eer)
print("EER Threshold: ", eer_threshold)

# # Display some results
# print("Hamming Distances:", hamming_distances)
# print("Labels:", labels)
