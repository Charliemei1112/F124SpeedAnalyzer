# F1 Speed Analyzer - Main Processing Script
# Modified for web application integration

import sys
import os
from frames import extract_frames
from evaluation import load_model, predict_speed
import matplotlib.pyplot as plt
import json
import shutil
import numpy as np
import traceback
from datetime import datetime
import time
import psutil  # For memory monitoring
import gc  # For garbage collection

def print_progress(message):
    """Print progress messages with timestamp for the web interface"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")
    sys.stdout.flush()  # Ensure immediate output

def print_memory_usage(stage=""):
    """Print current memory usage"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / 1024 / 1024  # Convert to MB
    memory_percent = process.memory_percent()
    
    if stage:
        print_progress(f"Memory Usage ({stage}): {memory_mb:.1f} MB ({memory_percent:.1f}% of system)")
    else:
        print_progress(f"Memory Usage: {memory_mb:.1f} MB ({memory_percent:.1f}% of system)")

def main():
    try:
        # Check command line arguments
        if len(sys.argv) != 2:
            print_progress("ERROR: Invalid number of arguments. Usage: python main.py <video_file>")
            sys.exit(1)

        # Get the video file path
        video_file = sys.argv[1]
        print_progress(f"Starting analysis of video: {os.path.basename(video_file)}")
        print_memory_usage("Startup")

        # Check if the file exists
        if not os.path.exists(video_file):
            print_progress(f"ERROR: File {video_file} does not exist")
            sys.exit(1)

        # Validate file format
        valid_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        file_ext = os.path.splitext(video_file)[1].lower()
        if file_ext not in valid_extensions:
            print_progress(f"ERROR: Unsupported file format {file_ext}. Supported: {', '.join(valid_extensions)}")
            sys.exit(1)

        print_progress("Starting video frame extraction and speed analysis...")
        print_memory_usage("Before frame extraction")
        
        start_time = time.time()
        # Extract frames and analyze speed
        speed_data = extract_frames(video_file)
        end_time = time.time()
        print_progress(f"Time taken: {round(end_time - start_time, 3)} seconds")
        print_memory_usage("After frame extraction")
        
        if not speed_data or len(speed_data) == 0:
            print_progress("ERROR: No speed data extracted from video")
            sys.exit(1)

        print_progress(f"Successfully extracted {len(speed_data)} speed data points")
        print_progress("Calculating performance metrics...")
        
        # Generate visualization
        # print_progress("Generating speed chart...")
        plt.figure(figsize=(12, 6))
        plt.plot(speed_data, linewidth=2, color='#667eea')
        plt.title('F1 Speed Analysis Results', fontsize=16, fontweight='bold')
        plt.xlabel('Frame Number', fontsize=12)
        plt.ylabel('Speed (km/h)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save the plot
        plt.savefig('speed.png', dpi=300, bbox_inches='tight')
        plt.close()  # Close the figure to free memory
        # print_progress("Speed chart saved as speed.png")

        # Save the speed data to JSON file
        # print_progress("Saving speed data to JSON...")
        with open('speed.json', 'w') as f:
            json.dump(speed_data, f, indent=2)
        # print_progress("Speed data saved as speed.json")

        # Create a .dat file for compatibility
        # print_progress("Creating data file...")
        with open('speed.dat', 'w') as f:
            for i, speed in enumerate(speed_data):
                f.write(f"{speed}\n")
        # print_progress("Data file saved as speed.dat")

        # Convert to numpy array for metrics calculation
        speed_array = np.array(speed_data)
        
        # Calculate comprehensive metrics
        metrics = {
            'video_file': os.path.basename(video_file),
            'analysis_timestamp': datetime.now().isoformat(),
            'total_frames': len(speed_data),
            'Mean Speed': float(np.mean(speed_array)),
            'Minimum Speed': int(np.min(speed_array)),
            'Maximum Speed': int(np.max(speed_array)),
            'Median Speed': float(np.median(speed_array)),
            'Speed Range': int(np.max(speed_array) - np.min(speed_array)),
            'Non-zero Frames': int(np.count_nonzero(speed_array))
        }

        # Save metrics to JSON file
        with open('metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        # print_progress("Metrics saved as metrics.json")

        # Create a comprehensive results file for easy download
        # print_progress("Creating comprehensive results file...")
        comprehensive_results = {
            'metadata': {
                'video_file': os.path.basename(video_file),
                'analysis_timestamp': datetime.now().isoformat(),
                'total_frames': len(speed_data),
                'processing_successful': True
            },
            'metrics': metrics,
            # 'speed_data': speed_data,
            'summary': {
                'average_speed': round(metrics['Mean Speed'], 2),
                'max_speed': metrics['Maximum Speed'],
                'min_speed': metrics['Minimum Speed'],
                'total_data_points': len(speed_data)
            }
        }
        
        with open('results.json', 'w') as f:
            json.dump(comprehensive_results, f, indent=2)
        # print_progress("Comprehensive results saved as results.json")

        # Final summary
        print_progress("ANALYSIS COMPLETE!")
        print_memory_usage("Final")
        print_progress("=" * 50)
        print_progress(f"Video: {os.path.basename(video_file)}")
        print_progress(f"Frames processed: {len(speed_data)}")
        print_progress(f"Average speed: {round(metrics['Mean Speed'], 1)} km/h")
        print_progress(f"Max speed: {metrics['Maximum Speed']} km/h")
        print_progress(f"Min speed: {metrics['Minimum Speed']} km/h")

    except KeyboardInterrupt:
        print_progress("Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print_progress(f"ERROR: Analysis failed with exception: {str(e)}")
        print_progress("Full error traceback:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()