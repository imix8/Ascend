# Ascend

<img width="1256" alt="Screenshot 2024-02-11 061220" src="https://github.com/imix8/Ascend/assets/112455598/0667cfdc-55f6-4c4e-8dd1-ceba6eda9268">

Ascend is designed to provide climbers with detailed insights into their climbing technique utilizing an object detection & image segmentation model (YOLOv8) from Ultralytics.  Furthermore, advanced edge detection algorithms, easy to navigate Taipy GUI and multiple methods of feedback makes this project truly unique. By uploading a video of a climbing session, users can obtain various analyses such as the climber's center of mass, utilized holds, and body posture throughout the climb. This README provides an overview of the project, including its features, installation instructions, usage, and technical architecture.

<img width="225" alt="pic1" src="https://github.com/imix8/Ascend/assets/112455598/3f7a2821-1ea3-4bfd-89b8-6aa518e18f00">

## Features

- **Video Upload**: Users can upload a .mp4 video of their climb to be processed.
- **Pose Estimation**: Analyzes the climber's posture and movements throughout the climb.
- **Center of Mass Calculation**: Identifies and tracks the climber's center of mass for balance analysis while providing feedback with colored visual queues.
- **Utilized Holds Detection**: Analyzes the holds utilized during the climb and draws bounding boxes to distinguish between used and unused holds.
- **Video Download**: Users can download all processed videos, which includes visual overlays of the analysis, to their local machine.

<img width="225" alt="pic2" src="https://github.com/imix8/Ascend/assets/112455598/c79c6b9a-5b86-48fa-bb79-38dd73174761">

### Prerequisites

- Python 3.11

### Dependencies

- Ultralytics
- OpenCV
- Taipy GUI
- scikit-learn

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/imix8/Ascend.git
   ```
2. Navigate to the cloned directory:
   ```bash
   cd Ascend
   ```
3. Install dependencies using requirements.txt:
   ```bash
   pip install requirements.txt
   ```

## Usage

1a. Start the application:
   ```bash
   python main.py
   ```
1b. An alternative is to go to the IP_ADDR with: "ascend-climbing.tech."
2. Open a web browser and navigate to `http://localhost:8000` to access the GUI.
3. Upload a climbing video in .mp4 format using the "Upload Climb Here!" button.
4. The analysis will begin automatically after the video is uploaded.  Progress notifications will appear in the lower left of the GUI to provide status.
5. Once the analysis is complete, downloading the processed videos and other output image will be possible using the "Download Climb Here!" from the sidebar.

## Technical Design Architecture

The project is structured around a single-page Taipy GUI application, with the backend processing powered by OpenCV for image and video analysis, and YOLOv8 from Ultralytics for pose estimation (implicitly used through model predictions).  Critical design decisions are as follows:

1. **Edge Detection**:  We used open-cv's edge detection algorithm to detect the edges of climbing holds.  The image was first rasterized into discrete colors to simplify the process of detecting climbing hold edges  and also to split up each colored hold into different image masks.

2. **Image masks**:  The image masks used edge detection once again to get the bounding boxes of the holds.  We could then detect whether or not a persons hand, foot, or limb was touching a climbing hold by checking if its position is inside the bounding box.
 
<img width="225" alt="pic4" src="https://github.com/imix8/Ascend/assets/112455598/a77da975-49a4-4acd-a0f4-d2e3032f405b">

## Development and Contribution

Contributions to the Climbing Analysis Tool are welcome! Whether it's feature requests, bug reports, or code contributions, please feel free to reach out or submit a pull request.

<img width="224" alt="pic3" src="https://github.com/imix8/Ascend/assets/112455598/0a35a541-e013-412f-842d-049277382188">

## License

MIT License

---
