# AI_VISION
**How to Run It Locally**

**1. Install the Requirements**
Open your terminal or command prompt and install the required dependencies:

``Bash
pip install ultralytics opencv-python numpy matplotlib``

**2. Set Up Your Project Folder**
Create a local folder (e.g., traffic_detector/) and place:

solution.py file inside it.

Your test videos (e.g., sample_01.mp4, sample_02.mp4) inside it.

**3. Run a Local Test Script**

Create a file named run_local.py in that same folder with the following code to execute the pipeline and generate your predictions or graphs:

```Python

import cv2
import json
from solution import detect_events, RiskEstimator

video_path = "sample_02.mp4"  # Change to your local video file name
print(f"Running Part A (Discrete Events) on {video_path}...")
events = detect_events(video_path)

print(f"\n--- DONE! Detected {len(events)} events ---")
for event in events:
    print(f"Start: {event[0]:.2f}s | End: {event[1]:.2f}s | Type: {event[2]}")

#Save results locally
with open("local_predictions.json", "w") as f:
    json.dump(events, f, indent=2)
print("Saved to local_predictions.json")
```
