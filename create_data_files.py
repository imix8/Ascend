from ultralytics import YOLO
import os
import numpy as np
import json

def process_video(file_name):
    model = YOLO('yolov8m-pose.pt')
    results = model(source=f'./pre_data/{file_name}', show=False, conf=0.3, save=True)
    path = f'./post_data/'
    dict = {}
    for i, r in enumerate(results):
        dict[str(i)] = json.loads(r.tojson())
    for key in dict:
        for item in dict[key]:
            keypoints = item['keypoints']
            x_coords = keypoints['x']
            y_coords = keypoints['y']
            x_mean = mean_exclude_zeros(x_coords)
            y_mean = mean_exclude_zeros(y_coords)
            keypoints['mean'] = {'x': x_mean, 'y': y_mean}
    with open(f'{path}/{file_name.split(".")[0]}_{len(dict)}.json', 'w') as file:
        json.dump(dict, file)

def mean_exclude_zeros(numbers):
    filtered_numbers = [num for num in numbers if num != 0]
    return np.mean(filtered_numbers)
    
if __name__ == '__main__':
    process_video("test_1.mp4")