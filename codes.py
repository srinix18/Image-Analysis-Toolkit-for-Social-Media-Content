#run on google colab 
#Hue saturation and contrast


import cv2
import numpy as np
import pandas as pd

def extract_hsv_contrast(image_path):
    """
    Extracts the HSV value, saturation, and contrast from an image.

    Args:
        image_path: Path to the image file.

    Returns:
        A tuple containing the hue, saturation, and contrast of the image.
    """

    image = cv2.imread(image_path)  # Read the image directly using OpenCV

    # Convert RGB image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Extract HSV value, saturation, and contrast
    hue = hsv_image[:, :, 0].mean()
    saturation = hsv_image[:, :, 1].mean()
    contrast = cv2.meanStdDev(image)[0]

    return hue, saturation, contrast

# Read image paths from the file
image_paths = []
with open("/content/drive/MyDrive/virtual influencer data/instagram_post_images/shudu.gram_extracted_paths.txt") as f:
    for line in f:
        image_paths.append(line.strip())

# Extract HSV value, saturation, and contrast for each image
data = []
for image_path in image_paths:
    hue, saturation, contrast = extract_hsv_contrast(image_path)
    data.append((image_path, hue, saturation, contrast))

# Create a Pandas DataFrame
df = pd.DataFrame(data, columns=['image_path', 'hue', 'saturation', 'contrast'])

# Save the DataFrame as a CSV file
df.to_csv('hsv_saturation_contrast.csv', index=False)

print("HSV Contrast data saved successfully!")

from google.colab import drive
drive.mount('/content/drive')

"""Face Ratio"""

import numpy as np
import dlib
import cv2
import pandas as pd

# Read image paths from text file
image_paths = []
with open('/content/drive/MyDrive/virtual influencer data/instagram_post_images/shudu.gram_extracted_paths.txt', 'r') as f:
    for line in f:
        image_paths.append(line.strip())

# Initialize facial detector
detector = dlib.get_frontal_face_detector()

# Create empty DataFrame to store results
results_df = pd.DataFrame(columns=['Image Path', 'Face Detection Status', 'Face Pixels Ratio'])

# Process each image
for image_path in image_paths:
    # Load image
    image = cv2.imread(image_path)

    # Convert image to RGB (required by dlib)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect faces in the image
    detections = detector(image)

    # Determine face detection status
    if detections:
        detection_status = "Face Detected"
    else:
        detection_status = "Face Not Detected or obstruction"

    # Initialize face pixels and total pixels counters
    face_pixels = 0
    total_pixels = np.prod(image.shape[:2])

    # Iterate through detected faces
    for face in detections:
        # Extract face coordinates
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()

        # Crop face region
        cropped_face = image[y1:y2, x1:x2]

        # Count face pixels
        face_pixels += np.count_nonzero(cropped_face)

    # Calculate face-to-image pixel ratio
    face_to_image_ratio = face_pixels / total_pixels if detection_status == "Face Detected" else 0

    # Append result to DataFrame
    results_df = results_df.append({'Image Path': image_path, 'Face Detection Status': detection_status, 'Face Pixels Ratio': face_to_image_ratio}, ignore_index=True)

# Save results to CSV file
results_df.to_csv('FaceRatio.csv', index=False)

"""Face position"""

import dlib
import cv2
import pandas as pd

# Read image paths from text file
image_paths = []
with open('/content/drive/MyDrive/virtual influencer data/instagram_post_images/shudu.gram_extracted_paths.txt', 'r') as f:
    for line in f:
        image_paths.append(line.strip())

# Initialize facial detector
detector = dlib.get_frontal_face_detector()

# Initialize empty DataFrame to store results
results_df = pd.DataFrame(columns=['Image Path', 'Detection Status', 'Quadrant'])

# Process each image
for image_path in image_paths:
    # Load image
    image = cv2.imread(image_path)

    # Detect faces in the image
    detections = detector(image)

    # Determine detection status
    if detections:
        detection_status = "Face Detected"
    else:
        detection_status = "Face Not Detected"

    # Determine quadrant
    quadrant = None

    if detection_status == "Face Detected":
        # Extract face coordinates
        face = detections[0]  # Assume only one face is detected
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()

        # Determine image center
        image_center_x = image.shape[1] // 2
        image_center_y = image.shape[0] // 2

        # Determine quadrant based on coordinates
        if x1 > image_center_x and y1 < image_center_y:
            quadrant = "Upper-Right"
        elif x1 < image_center_x and y1 < image_center_y:
            quadrant = "Upper-Left"
        elif x1 < image_center_x and y1 > image_center_y:
            quadrant = "Lower-Left"
        elif x1 > image_center_x and y1 > image_center_y:
            quadrant = "Lower-Right"
        else:
            # Center-aligned face
            quadrant = "Center"
    else:
        # No face detected
        quadrant = "Face Not detectable or obstructed"

    # Append result to DataFrame
    results_df = results_df.append({'Image Path': image_path, 'Detection Status': detection_status, 'Quadrant': quadrant}, ignore_index=True)

# Save results to CSV file
results_df.to_csv('face_detection_quadrants.csv', index=False)

"""Emotion detection"""

import csv
from google.colab import drive
from keras.models import model_from_json
import cv2
import numpy as np



# Load the pre-trained model architecture from Google Drive
json_path = '/content/drive/MyDrive/fer.json'
with open(json_path, 'r') as json_file:
    loaded_model_json = json_file.read()
model = model_from_json(loaded_model_json)

# Load the pre-trained model weights from Google Drive
weights_path = '/content/drive/MyDrive/fer.h5'
model.load_weights(weights_path)

# Read image paths from a text file
with open('/content/drive/MyDrive/virtual influencer data/instagram_post_images/shudu.gram_extracted_paths.txt', 'r') as file:
    image_paths = file.read().splitlines()

# Create a CSV file to store the results
csv_file_path = '/content/emotion.csv'
with open(csv_file_path, 'w', newline='') as csvfile:
    fieldnames = ['Image', 'Predicted Emotion']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    # Write the header
    writer.writeheader()

    # Process each image
    for image_path in image_paths:
        # Load the image
        image = cv2.imread(image_path)

        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Resize the image to the model's input size
        resized_image = cv2.resize(gray_image, (48, 48))

        # Preprocess the image
        preprocessed_image = resized_image / 255.0
        preprocessed_image = np.expand_dims(preprocessed_image, axis=0)
        preprocessed_image = np.expand_dims(preprocessed_image, axis=-1)

        # Predict the emotion
        prediction = model.predict(preprocessed_image)

        # Get the emotion label
        emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        predicted_emotion = emotion_labels[np.argmax(prediction)]

        # Write the results to the CSV file
        writer.writerow({'Image': image_path, 'Predicted Emotion': predicted_emotion})

        # Print the predicted emotion for each image
        print(f'Image: {image_path}, Predicted Emotion: {predicted_emotion}')

print(f'Results saved to {csv_file_path}')

"""presence of multiple faces"""

!pip install mtcnn

import cv2
import mtcnn
import pandas as pd

# Load the MTCNN face detector
detector = mtcnn.MTCNN()

# Load image paths from text file
with open('/content/drive/MyDrive/virtual influencer data/instagram_post_images/shudu.gram_extracted_paths.txt', 'r') as f:
    image_paths = f.read().splitlines()

# Create a DataFrame to store results
df = pd.DataFrame(columns=["Image Path", "Multiple Faces"])

for image_path in image_paths:
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Error reading image: {image_path}")

        # Detect faces using MTCNN
        results = detector.detect_faces(image)

        # Assign "Present" or "Not Present" directly to multiple_faces
        multiple_faces = "Present" if len(results) > 1 else "Not Present"

        # Store results in DataFrame
        df.loc[len(df)] = [image_path, multiple_faces]

    except Exception as e:
        print(f"Error processing image: {image_path} - {e}")

# Save DataFrame to Excel
df.to_excel('multiplefaces.xlsx', index=False)

"""to find the range of contrast"""

import pandas as pd
import ast
import re

def clean_description(description):
    # Convert to string and then remove emojis and special characters, keep only alphabets, numbers, and spaces
    cleaned_description = re.sub(r'[^a-zA-Z0-9\s]', '', str(description))
    return cleaned_description

def replace_and_clean_columns(file_path):
    # Load the Excel file into a DataFrame
    df = pd.read_excel(file_path)

    # Clean the 4th column (Description) by removing emojis and special characters
    df.iloc[:, 3] = df.iloc[:, 3].apply(clean_description)

    # Drop the 10th column
    df = df.drop(df.columns[9], axis=1)

    # Save the modified DataFrame to a new Excel file
    df.to_excel('modified_data.xlsx', index=False)

# Apply the code to the sample Excel file
excel_file_path = '/content/shudugram compiled result final (1).xlsx'
replace_and_clean_columns(excel_file_path)

# Display the modified content
modified_df = pd.read_excel('modified_data.xlsx')

"""days between each post and the latest post"""

import csv
import datetime

def calculate_days_from_first_post_and_write_csv(filename):
    """Calculates days between each post and the first post, writes results to a CSV file."""

    output_filename = "days_from_first_post_lil.csv"  # Customizable output filename
    with open(filename, 'r') as input_file, open(output_filename, 'w', newline='') as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["date", "time", "days_from_first_post"])  # Write header row

        first_post_date = None
        for line in input_file:
            date_str, time_str = line.strip().split()
            posting_date = datetime.datetime.strptime(date_str, "%Y-%m-%d")  # Parse date (adjust format if needed)

            if first_post_date is None:
                first_post_date = posting_date
            else:
                days_from_first_post = (first_post_date-posting_date).days
                writer.writerow([date_str, time_str, days_from_first_post])  # Write data to CSV

# Replace with your actual filename
calculate_days_from_first_post_and_write_csv("/content/blah2.txt")

"""to extract file paths from directory"""

import os
# Replace "/path/to/your/directory" with the actual directory path on your Drive
directory_path = "/content/drive/MyDrive/virtual influencer data/instagram_post_images/shudu.gram"

# Initialize an empty list to store paths
jpg_paths = []

# Walk through the directory and subdirectories
for root, _, files in os.walk(directory_path):
    for filename in files:
        if filename.endswith(".jpg"):
            # Construct full paths and add to the list
            jpg_paths.append(os.path.join(root, filename))

# Open a new text file for writing
with open("shudu.gram_extracted_paths.txt", "w") as f:
    # Write each path on a separate line
    for path in jpg_paths:
        f.write(path + "\n")

print(f"{len(jpg_paths)} .jpg paths extracted and saved to extracted_paths.txt")

"""if the image is a part of a post with multiple images or single images"""

import re
import csv

# Specify the input and output file paths
input_file = "image_paths.txt"
output_csv = "results.csv"

# Open the input text file and create the CSV writer
with open("/content/blah.txt", "r") as infile, open("results_lilmiquela.csv", "w", newline="") as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(["path", "single/multiple"])

    # Process each line in the input file
    for line in infile:
        path = line.strip()

        # Check for single image pattern
        if path.endswith("UTC.jpg"):
            csv_writer.writerow([path, "single"])

        # Check for multiple image pattern
        elif re.match(r".*UTC_(\d+)\.jpg$", path):
            n = re.search(r"UTC_(\d+)\.jpg$", path).group(1)
            csv_writer.writerow([path, f"multiple ({n})"])

        else:
          print(line)

print("CSV file generated successfully!")

"""Clustering"""

import cv2
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd

# Function to read image paths from a text file
def read_image_paths(file_path):
    with open(file_path, 'r') as file:
        image_paths = [line.strip() for line in file]
    return image_paths

# Function to read and preprocess images
def preprocess_images(image_paths):
    images = [cv2.imread(img_path) for img_path in image_paths]
    # Assuming images are all of the same size, resize if needed
    resized_images = [cv2.resize(img, (100, 100)) for img in images]
    flattened_images = [img.flatten() for img in resized_images]
    return np.array(flattened_images)

# Function to perform K-means clustering
def perform_clustering(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_assignments = kmeans.fit_predict(data)
    return cluster_assignments

# Function to save clustered image paths to CSV
def save_clustered_paths_to_csv(image_paths, cluster_assignments, csv_file_path):
    df = pd.DataFrame({'Image_Path': image_paths, 'Cluster': cluster_assignments})
    df.to_csv(csv_file_path, index=False)

# Example usage
text_file_path = '/content/blah lil.txt'  # Provide the path to your text file with image paths
n_clusters = 5  # Specify the number of clusters
csv_file_path = 'clustered_image_paths.csv'  # Specify the CSV file path

# Read image paths from the text file
image_paths = read_image_paths(text_file_path)

# Preprocess images
data = preprocess_images(image_paths)

# Perform clustering
cluster_assignments = perform_clustering(data, n_clusters)

# Save clustered image paths to CSV
save_clustered_paths_to_csv(image_paths, cluster_assignments, csv_file_path)

"""symmetry"""

!pip install scikit-image

import cv2
import pandas as pd
from skimage.metrics import structural_similarity as ssim

# Read Excel File
excel_file_path = '/content/Combined result.xlsx'
df = pd.read_excel(excel_file_path)

# Create a list to store symmetry results
symmetry_results = []

# Iterate through the rows of the DataFrame
for index, row in df.iterrows():
    image_path = row['image_path']  # Replace with the actual column name
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Compute the horizontal flip of the image
    flipped_image = cv2.flip(gray_image, 1)

    # Calculate structural similarity index
    ssim_index, _ = ssim(gray_image, flipped_image, full=True)

    # If the SSIM index is close to 1, it indicates high symmetry
    symmetry_score = ssim_index

    # Append result to the list
    symmetry_results.append({'Image_Path': image_path, 'Symmetry_Score': symmetry_score})

# Convert the results list to a DataFrame
symmetry_results_df = pd.DataFrame(symmetry_results)

# Save the DataFrame to a CSV file
symmetry_results_csv_path = '/content/symmetry_results.csv'
symmetry_results_df.to_csv(symmetry_results_csv_path, index=False)

print(f"Symmetry results saved to {symmetry_results_csv_path}")

"""text in image"""

!pip install easyocr

"""visual balance"""

import cv2
import pandas as pd
import numpy as np

# Read Excel File
excel_file_path = '/content/Combined result.xlsx'
df = pd.read_excel(excel_file_path)

# Create a list to store visual balance color results
visual_balance_results = []

# Iterate through the rows of the DataFrame
for index, row in df.iterrows():
    image_path = row['image_path']  # Replace with the actual column name
    image = cv2.imread(image_path)

    # Split the image into left and right halves
    height, width, _ = image.shape
    left_half = image[:, :width // 2, :]
    right_half = image[:, width // 2:, :]

    # Calculate mean color values for left and right halves
    mean_color_left = np.mean(left_half, axis=(0, 1))
    mean_color_right = np.mean(right_half, axis=(0, 1))

    # Check if the mean color values are similar (within a threshold)
    color_similarity_threshold = 30  # Adjust the threshold based on your dataset
    visual_balance_color = np.all(np.abs(mean_color_left - mean_color_right) < color_similarity_threshold)

    # Append result to the list
    visual_balance_results.append({'Image_Path': image_path, 'Visual_Balance_Color': visual_balance_color})

# Convert the results list to a DataFrame
visual_balance_results_df = pd.DataFrame(visual_balance_results)

# Save the DataFrame to a CSV file
visual_balance_results_csv_path = '/content/visual_balance_results.csv'
visual_balance_results_df.to_csv(visual_balance_results_csv_path, index=False)

print(f"Visual balance color results saved to {visual_balance_results_csv_path}")

"""diagonal dominance"""

import cv2
import pandas as pd
import numpy as np

# Read Excel File
excel_file_path = '/content/Combined result.xlsx'
df = pd.read_excel(excel_file_path)

# Create a list to store diagonal dominance results
diagonal_dominance_results = []

# Iterate through the rows of the DataFrame
for index, row in df.iterrows():
    image_path = row['image_path']  # Replace with the actual column name
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Compute the intensity profile along the main diagonal
    diagonal_profile = np.diag(image)

    # Calculate the standard deviation of the intensity values along the diagonal
    diagonal_std = np.std(diagonal_profile)

    # If the standard deviation is above a threshold, it indicates diagonal dominance
    threshold = 20  # Adjust the threshold based on your dataset
    diagonal_dominant = diagonal_std > threshold

    # Append result to the list
    diagonal_dominance_results.append({'Image_Path': image_path, 'Diagonal_Dominance': diagonal_dominant})

# Convert the results list to a DataFrame
diagonal_dominance_results_df = pd.DataFrame(diagonal_dominance_results)

# Save the DataFrame to a CSV file
diagonal_dominance_results_csv_path = '/content/diagonal_dominance_results.csv'
diagonal_dominance_results_df.to_csv(diagonal_dominance_results_csv_path, index=False)

print(f"Diagonal dominance results saved to {diagonal_dominance_results_csv_path}")

"""mind  perception of the description"""

import pandas as pd
from transformers import pipeline

# Load the MindMiner model
model_name = "j-hartmann/MindMiner"
mindminer = pipeline(model=model_name, function_to_apply="none", device=-1)

# Read the Excel file
excel_file_path = '/content/Combined result.xlsx'
df = pd.read_excel(excel_file_path)

# Extract the 'Description' column as a list of strings
texts = df['Description'].astype(str).tolist()

# Apply mindminer to the list of texts
results = [mindminer(text) for text in texts]

# Add the results to your DataFrame
df['MindMiner_Predictions'] = results

# Display or save the results
print(df[['Description', 'MindMiner_Predictions']])
df.to_excel('/content/output_with_mindminer_predictions.xlsx', index=False)

"""sentiment of the description"""

!pip install pysentimiento

import pandas as pd
from pysentimiento import create_analyzer

# Read the Excel file
excel_file_path = '/content/Combined result.xlsx'
df = pd.read_excel(excel_file_path)

# Create a sentiment analysis analyzer for Spanish
analyzer = create_analyzer(task="sentiment", lang="en")

# Extract the 'Description' column as a list of strings
texts = df['Description'].astype(str).tolist()

# Apply sentiment analysis to the list of texts
results = [analyzer.predict(text).output for text in texts]

# Add the results to your DataFrame
df['Sentiment_Predictions'] = results

# Display or save the results
print(df[['Description', 'Sentiment_Predictions']])
df.to_excel('/content/output_with_sentiment_predictions.xlsx', index=False)
