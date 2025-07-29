import cv2
import numpy as np
import re
from PIL import Image
import os
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

# set 3 images for 3 digits
# hundreds digit box
left_hundreds = 0.531
right_hundreds = 0.538
top_hundreds = 0.913
bottom_hundreds = 0.931

# tens digit box
left_tens = 0.538
right_tens = 0.5445

# ones digit box
left_ones = 0.544
right_ones = 0.551

# shared height region
top = 0.914
bottom = 0.930

def extract_speed_image(image):
    # crop the image for hundreds digit
    digit_0 = image[int(image.shape[0] * top):int(image.shape[0] * bottom), 
                    int(image.shape[1] * left_hundreds):int(image.shape[1] * right_hundreds)]
    
    # crop the image for tens digit
    digit_1 = image[int(image.shape[0] * top):int(image.shape[0] * bottom), 
                    int(image.shape[1] * left_tens):int(image.shape[1] * right_tens)]
    
    # crop the image for ones digit
    digit_2 = image[int(image.shape[0] * top):int(image.shape[0] * bottom), 
                    int(image.shape[1] * left_ones):int(image.shape[1] * right_ones)]
    
    # convert to grayscale
    digit_0 = cv2.cvtColor(digit_0, cv2.COLOR_BGR2GRAY)
    digit_1 = cv2.cvtColor(digit_1, cv2.COLOR_BGR2GRAY)
    digit_2 = cv2.cvtColor(digit_2, cv2.COLOR_BGR2GRAY)

    # enhance the image by thresholding and resizing with scaling factor:2
    _, digit_0 = cv2.threshold(digit_0, 225, 255, cv2.THRESH_BINARY)    
    _, digit_1 = cv2.threshold(digit_1, 225, 255, cv2.THRESH_BINARY)    
    _, digit_2 = cv2.threshold(digit_2, 225, 255, cv2.THRESH_BINARY)    
    digit_0 = cv2.resize(digit_0, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    digit_1 = cv2.resize(digit_1, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    digit_2 = cv2.resize(digit_2, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # apply morphological operations
    kernel = np.ones((2,2), np.uint8)
    digit_0 = cv2.morphologyEx(digit_0, cv2.MORPH_CLOSE, kernel)
    digit_1 = cv2.morphologyEx(digit_1, cv2.MORPH_CLOSE, kernel)
    digit_2 = cv2.morphologyEx(digit_2, cv2.MORPH_CLOSE, kernel)

    # Apply final binary thresholding to ensure only 0 and 255 values
    _, digit_0 = cv2.threshold(digit_0, 127, 255, cv2.THRESH_BINARY)
    _, digit_1 = cv2.threshold(digit_1, 127, 255, cv2.THRESH_BINARY)
    _, digit_2 = cv2.threshold(digit_2, 127, 255, cv2.THRESH_BINARY)

    # resize the images to 26x40
    digit_0 = cv2.resize(digit_0, (26, 40))
    digit_1 = cv2.resize(digit_1, (26, 40))
    digit_2 = cv2.resize(digit_2, (26, 40))

    return [digit_0, digit_1, digit_2]

def process_single_frame(args):
    """Process a single frame - helper function for parallel processing"""
    frame_path, frame_name, output_dir, frame_index = args
    
    # extract the speed image
    speed_images = extract_speed_image(f"{frame_path}/{frame_name}")
    # save the speed images for each digit
    cv2.imwrite(f"{output_dir}/digit_{frame_index*3:04d}.png", speed_images[0])
    cv2.imwrite(f"{output_dir}/digit_{frame_index*3+1:04d}.png", speed_images[1])
    cv2.imwrite(f"{output_dir}/digit_{frame_index*3+2:04d}.png", speed_images[2])
    
    return frame_index

def extract_speed_images(frame_path):
    # Detect number of CPU cores and use half of them
    cpu_count = os.cpu_count()
    max_workers = max(1, cpu_count // 1)  # Use half the cores, minimum 1
    print(f"Detected {cpu_count} CPU cores, using {max_workers} workers for parallel processing")
    
    # create a folder to save the speed images
    output_dir = f"{frame_path}_speed_images"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # prepare arguments for parallel processing
    frames = sorted(os.listdir(frame_path))
    args_list = [(frame_path, frame, output_dir, i) for i, frame in enumerate(frames)]
    
    # process frames in parallel using half of CPU cores
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # use tqdm to show progress
        list(tqdm(executor.map(process_single_frame, args_list), 
                 total=len(frames), desc="Extracting speed images"))
    
def main():
    # ask for user input
    frame_path = input("Enter the path to the frames: ")
    # check if the path exists
    if not os.path.exists(frame_path):
        print("Path does not exist")
        exit()
    extract_speed_images(frame_path)

if __name__ == "__main__":
    main()
    # # test
    # digits = extract_speed_image("Haas/frame_0277.png")
    # # save the digits
    # cv2.imwrite("digits_0.png", digits[0])
    # cv2.imwrite("digits_1.png", digits[1])
    # cv2.imwrite("digits_2.png", digits[2])
