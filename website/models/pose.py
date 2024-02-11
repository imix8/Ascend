from ultralytics import YOLO

def predict_pose(video_path: str):
    model = YOLO('yolov8m-pose.pt')

    results = model(source=video_path, stream=True, conf=0.3, save=True)
    return results