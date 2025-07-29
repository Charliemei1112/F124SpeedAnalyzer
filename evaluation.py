"""
    Using the model saved in digit_cnn_model.pth, 
    predict the labels for the unknown images
"""

from preprocess import image_to_array
from train import DigitCNN  # Import the model from train_cnn.py
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import json
import csv
from tqdm import tqdm
from PIL import Image

def load_model():
    """Load the trained model and return it with label mapping"""
    # Load label mapping
    with open('model/label.json', 'r') as f:
        label_mapping = json.load(f)
    
    # Create model
    num_classes = len(label_mapping['idx_to_label'])
    model = DigitCNN(num_classes)
    
    # Load weights
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load('model/model.pth', map_location=device))
    model.to(device)
    model.eval()
    
    return model, label_mapping, device

def predict_digits(model, label_mapping, device, test_data):
    """
    Given a list of 130-byte packed data, return list of predictions
    
    Args:
        model: trained CNN model
        label_mapping: dictionary mapping indices to labels
        device: torch device
        test_data: list of 130-byte numpy arrays (packed bits)
    
    Returns:
        list of predicted labels (strings: '0', '1', ..., '9', '*')
    """
    predictions = []
    
    for packed_data in test_data:
        # Unpack the 130-byte packed data back to 26x40 binary array
        unpacked = np.unpackbits(packed_data)
        
        # Reshape to 26x40 (trim padding from packbits)
        binary_array = unpacked[:26*40].reshape(26, 40)
        
        # Convert to tensor: add batch and channel dimensions
        tensor = torch.FloatTensor(binary_array).unsqueeze(0).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            output = model(tensor)
            _, predicted_idx = torch.max(output, 1)
            predicted_label = label_mapping['idx_to_label'][str(predicted_idx.item())]
            predictions.append(predicted_label)
    
    return predictions

def digit_to_speed(digit_0, digit_1, digit_2, model, label_mapping, device):
    """
    Convert three digits to speed
    """

    # convert cv2 into PIL image
    digit_0 = Image.fromarray(digit_0)
    digit_1 = Image.fromarray(digit_1)
    digit_2 = Image.fromarray(digit_2)

    # convert the digits to array
    digit_0 = image_to_array(digit_0)
    digit_1 = image_to_array(digit_1)
    digit_2 = image_to_array(digit_2)

    # unpack the digits
    digit_0 = np.unpackbits(digit_0)
    digit_1 = np.unpackbits(digit_1)
    digit_2 = np.unpackbits(digit_2)

    # reshape the digits
    digit_0 = digit_0[:26*40].reshape(26, 40)
    digit_1 = digit_1[:26*40].reshape(26, 40)
    digit_2 = digit_2[:26*40].reshape(26, 40)

    # convert the digits to tensor
    digit_0 = torch.FloatTensor(digit_0).unsqueeze(0).unsqueeze(0).to(device)
    digit_1 = torch.FloatTensor(digit_1).unsqueeze(0).unsqueeze(0).to(device)
    digit_2 = torch.FloatTensor(digit_2).unsqueeze(0).unsqueeze(0).to(device)

    digits = []

    # predict the digits
    with torch.no_grad():
        output_0 = model(digit_0)
        output_1 = model(digit_1)
        output_2 = model(digit_2)

        # get the predicted digits
        _, idx_0 = torch.max(output_0, 1)
        _, idx_1 = torch.max(output_1, 1)
        _, idx_2 = torch.max(output_2, 1)
        
        # get the predicted labels
        label_0 = label_mapping['idx_to_label'][str(idx_0.item())]
        label_1 = label_mapping['idx_to_label'][str(idx_1.item())]
        label_2 = label_mapping['idx_to_label'][str(idx_2.item())]

        # append the predicted digits
        digits.append(label_0)
        digits.append(label_1)
        digits.append(label_2)

    # convert the digits to speed
    speed = 0
    for digit in digits:
        if digit == '*':
            continue
        else:
            speed = 10 * speed + int(digit)

    return speed

def predict_speed(video_dir, model, label_mapping, device):
    # Convert each image into nparray with image_to_array
    test_data = load_images(video_dir)
    print(f"Loaded {len(test_data)} images")

    # Predict the labels for the test data
    predictions = predict_digits(model, label_mapping, device, test_data)
    
    # Convert predictions to speed readings (groups of 3)
    speed_data = []
    
    # apply tqdm to the loop
    for i in tqdm(range(0, len(predictions), 3), desc="Converting predictions to speed readings"):
        speed = 0
        for j in range(3):
            if predictions[i+j] == '*':
                continue
            else:
                speed = 10 * speed + int(predictions[i+j])
        speed_data.append(speed)
    
    return speed_data

def load_images(image_dir):
    """Load all images from directory as packed data"""
    images = [f for f in os.listdir(image_dir) if f.endswith('.png')]
    images.sort()
    
    test_data = []
    for image in images:
        test_data.append(image_to_array(os.path.join(image_dir, image)))
    
    return test_data

def main():
    # Get test directory
    test_dir = input("Enter the path to the test directory: ")
    if not os.path.exists(test_dir):
        print("Directory does not exist")
        return
    
    # Load the model
    model, label_mapping, device = load_model()

    
    speed_data = predict_speed(test_dir, model, label_mapping, device)
    print(f"Extracted {len(speed_data)} speed readings")
    
    # Plot the speed data
    plt.plot(speed_data)
    plt.title('Speed Data')
    plt.xlabel('Reading Number')
    plt.ylabel('Speed')
    # plt.show()
    plt.savefig('speed_data.png')

    # Save the speed data to a csv file
    with open('speed_data.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['speed'])  # Header
        for speed in speed_data:
            writer.writerow([speed])
    
    print("Speed data saved to speed_data.csv")

if __name__ == "__main__":
    main()