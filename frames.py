import cv2
import os
import shutil
from tqdm import tqdm
from speed import extract_speed_image
from preprocess import image_to_array
from PIL import Image
from evaluation import load_model, digit_to_speed
import time
from datetime import datetime
import multiprocessing as mp
import sys
import psutil

model, label_mapping, device = load_model()

def print_memory_usage(stage=""):
    """Print current memory usage"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / 1024 / 1024  # Convert to MB
    memory_percent = process.memory_percent()
    

def print_progress(message):
    """Print progress messages with timestamp for the web interface"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")
    sys.stdout.flush()  # Ensure immediate output

def frame_to_speed(frame):
    # call extract_speed_image function to create digit images
    digit_0, digit_1, digit_2 = extract_speed_image(frame)
    # append the speed data
    speed = digit_to_speed(digit_0, digit_1, digit_2, model, label_mapping, device)
    return speed

def extract_frames(video_path):
    # open the video file
    cap = cv2.VideoCapture(video_path)  
    if not cap.isOpened():
        print("Error: Could not open video file")
        return
    
    # print_memory_usage("Before frame extraction")

    # get the frame count and other metadata
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)

    # print the video dimensions
    print_progress(f"Resolution: {width}x{height}")
    
    # Find the actual number of frames
    actual_frame_count = 0
    cap = cv2.VideoCapture(video_path)
    speed_data = []



    # add progress bar for this
    # add timestamp to the progress bar
    with tqdm(total=frame_count, desc=f"[{datetime.now().strftime('%H:%M:%S')}] Processing Frames:") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            speed_data.append(frame_to_speed(frame))
            actual_frame_count += 1
            pbar.update(1)
    cap.release()

    # print_progress(f"Frames processed: {actual_frame_count}")
    
    # if frame_count != actual_frame_count:
    #     # print(f"WARNING: Frame count mismatch! Using actual count: {actual_frame_count}")
    #     frame_count = actual_frame_count
    
    # # Reopen video for extraction
    # cap = cv2.VideoCapture(video_path)

    # speed_data = []

    # with tqdm(total=frame_count, desc="Generating Speed Data") as pbar:
    #     for i in range(frame_count):
    #         ret, frame = cap.read()
    #         speed_data.append(frame_to_speed(frame))
    #         pbar.update(1)
    
    cap.release()
    return speed_data

if __name__ == "__main__":
    # ask for user input
    video_path = input("Enter the path to the video file: ")
    # check if the video path is valid
    if not os.path.exists(f"{video_path}.mp4"):
        print("Invalid video path")
        exit()
    extract_frames(video_path)