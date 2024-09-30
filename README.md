
# Hand Gesture Recognition with OpenCV and MediaPipe

This project uses OpenCV and MediaPipe to detect hand gestures from a live camera based on the distance between the user's hands.

## Features

- Detects hand landmarks using MediaPipe.
- Calculates the distance between two hands.
- Dynamically displays images based on the proximity of the hands
- Images are resized proportionally to the distance between the hands.
- The project supports images with transparency (alpha channel).

## Requirements

To run this project, you need the following libraries:

```bash
pip install opencv-python mediapipe numpy
```
    
## How to Run

- Clone the repository and navigate to the project directory.
- Run the script:

```bash
  python main.py
```
