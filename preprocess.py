import os
from PIL import Image
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

def image_to_array(img):
    '''
    Given the image path, return the image as a 26*40 numpy array
    Notice that the value is threhsold so it can only be 0 or 255
    Use binary to represent the array so that it reduces the memory usage to 1/8
    '''
    # img = Image.open(image_path)
    img_array = np.array(img)

    # change all 255 to 1 and convert to bool (still 1 byte per element but cleaner)
    img_array = np.where(img_array == 255, 1, img_array).astype(np.uint8)

    # pack the array into 1 byte per pixel
    img_array = np.packbits(img_array)
    return img_array

def load_data(image_dir, validation_size=0.1, test_size=0.1):
    images = [f for f in os.listdir(image_dir)]
    images.sort()  # Sort to match label order
    print(f"Total images: {len(images)}")

    data = []
    labels = []

    # Load labels
    with open('digits.json', 'r') as f:
        labels = json.load(f)

    # Load data
    for image in images:
        img_array = image_to_array(os.path.join(image_dir, image))
        data.append(img_array)

    # shuffle data and labels, make sure the order is the same
    data, labels = shuffle(data, labels, random_state=42)

    validation_start = int(len(data) * (1 - test_size - validation_size))
    test_start = int(len(data) * (1 - test_size))

    train_data = data[:validation_start]
    train_labels = labels[:validation_start]
    validation_data = data[validation_start:test_start]
    validation_labels = labels[validation_start:test_start]
    test_data = data[test_start:]
    test_labels = labels[test_start:]

    return train_data, validation_data, test_data, train_labels, validation_labels, test_labels


def main():
    data_splits = load_data('Haas_speed_images', validation_size=0.1, test_size=0.1)
    train_data, validation_data, test_data, train_labels, validation_labels, test_labels = data_splits

    
if __name__ == "__main__":
    main()

    