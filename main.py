"""
A single-page Taipy application.

Please refer to https://docs.taipy.io/en/latest/manuals/gui/ for more details.
"""

from PIL import Image
from taipy.gui import Gui, Markdown, notify
from pages.root import *
from models.pose import *
import torch
import numpy as np
import os
import cv2
from models.edge_detection import *



home_md = """<|toggle|theme|>

<page|layout|columns=350px 1fr|
<|sidebar|

<|./images/logo.png|image|>

<br/>
### Analyze your **Climb**{: .color-primary} from a .mp4 video

<br/>
Video Upload
<|{video_path}|file_selector|on_action=upload_video|extensions=.mp4|label=Upload Climb Here!|>

<br/>
Video Download
<|file_download|label=Download Climb Here!|active={fixed}|>
|>

<|container|
# **DATA**{: .color-primary}

Give it a try by uploading a video to witness the intricacies of your climb! You can download the processed video in full quality from the sidebar to view. 🧗🏻
<br/>

### Processing Video 📷 
<|{in_process}|image|>

### Center of Mass ⚖️
<|{com_img}|image|>

### Utilized Holds 🤙 
 <|{out_utilized_holds}|image|>

|>

|page>
"""

video_path = ""
in_process = ''
com_img = ''
out_utilized_holds = '' 
fixed = False
home = Markdown(home_md)

# function to overlay image 2 over image 1 with alpha
def overlay_image_alpha(img1, img2, alpha=0.5):
    img1 = img1.astype(float)
    img2 = img2.astype(float)
    weighted_sum = cv2.addWeighted(img1, 1, img2, 1 - alpha, 0)
    return weighted_sum

def plot_circle_at_xy(xy, image):
    x, y = xy
    cv2.circle(image, (x, y), 10, (0, 0, 255), -1)
    return image

def plot_line_between_xy(xy1, xy2, image, color):
    cv2.line(image, xy1, xy2, color, thickness=5)
    return image

def draw_square_at_bounding_box(xy, image, color):
    x, y, w, h = xy
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 5)
    return image

def upload_video(state):
    notify(state, 'info', 'Uploading original video...')
    notify(state, 'info', 'Processing climb...')

    n_colors = 3
    holds_image = get_first_frame(state.video_path)
    cv2.imwrite('base.jpg', holds_image)
    holds_image = apply_kmeans_image_tracing(holds_image, n_colors=n_colors)
    holds_image = apply_kmeans_image_tracing(draw_bounding_boxes_and_remove(holds_image),n_colors=n_colors,saturation_scale=1,value_scale=1)
    holds_image_masks = get_masks(holds_image)
    processed_boxes = []

    results = predict_pose(state.video_path)
    for i, r in enumerate(results):
        keypoints_coords = r.keypoints.xy
        x_col = keypoints_coords[0][:, :1]
        non_zero_x = x_col[x_col != 0]
        x_average = non_zero_x.mean()
        y_col = keypoints_coords[0][:, 1:]
        non_zero_y = y_col[y_col != 0]
        
        y_average = non_zero_y.mean()
        im_array = r.plot()
        im = Image.fromarray(im_array[..., ::-1])
        im.save(f'temp-{i}.jpg')
        img = cv2.imread(f'temp-{i}.jpg')

        base = cv2.imread("base.jpg")
        im_array = r.plot(img=base, boxes=False)
        im = Image.fromarray(im_array[..., ::-1])
        im.save(f'base-{i}.jpg')
        base = cv2.imread(f'base-{i}.jpg')
        if (non_zero_y.size(dim=0) > 1) and (non_zero_x.size(dim=0) == non_zero_y.size(dim=0)):
            maximum = np.argpartition(non_zero_y, -2)[-2:]
            # draw foot line
            xy1 = (int(non_zero_x[maximum][0]), int(non_zero_y[maximum][0]))
            xy2 = (int(non_zero_x[maximum][1]), int(non_zero_y[maximum][1]))
            base = plot_line_between_xy(xy1, xy2, base, (128, 128, 128))


        mask_num = 1
        bounding_boxes = find_bounding_boxes_from_mask(holds_image_masks[mask_num])
        if not (np.isnan(x_average) or np.isnan(y_average)):
            com = (int(x_average.item()), int(y_average.item()))
            img = plot_circle_at_xy(com, img)
            base = plot_circle_at_xy(com, base)

            # draw line from com to foot line
            if (non_zero_y.size(dim=0) > 1) and (non_zero_x.size(dim=0) == non_zero_y.size(dim=0)):
                xy2 = (int(x_average.item()), int(non_zero_y[maximum][0]))
                x_max = np.sort(non_zero_x[maximum])
                color = (0, 255, 0) if com[0] > x_max[0] and com[0] < x_max[1] else (0, 0, 255)
                base = plot_line_between_xy(com, xy2, base, color)
            
            for x in x_col:
                for y in y_col:
                    if not (np.isnan(x) or np.isnan(y)):
                        com = (int(x.item()), int(y.item()))
                        for box in bounding_boxes:
                            if com[0] > box[0] and com[0] < box[0] + box[2] and com[1] > box[1] and com[1] < box[1] + box[3]:
                                img = draw_square_at_bounding_box(box, img, (255, 255, 255))
                                if box not in processed_boxes:
                                    processed_boxes.append(box)
            cv2.imwrite(f'temp-{i}.jpg', img)
        
        cv2.imwrite(f'base-{i}.jpg', base)

        state.in_process = f'temp-{i}.jpg'
        state.com_img = f'base-{i}.jpg'
        if os.path.exists(f"temp-{i-1}.jpg"):
            os.remove(f"temp-{i-1}.jpg")
            os.remove(f"base-{i-1}.jpg")
            
    base = cv2.imread('base.jpg')
    all_boxes = find_bounding_boxes_from_mask(holds_image_masks[1])
    for box in all_boxes:
        if box not in processed_boxes:
            base = draw_square_at_bounding_box(box, base, (0, 0, 255))
        else:
            base = draw_square_at_bounding_box(box, base, (0, 255, 0))
    cv2.imwrite("base.jpg", base)

    state.out_utilized_holds = "base.jpg"

if __name__ == "__main__":
    pages = {
        "/": root_page,
        "home": home,
    }
    gui = Gui(pages=pages)
    gui.md = ""
    gui.run(title="Ascend", use_reloader=True, upload_folder="uploads/", port=80, host='0.0.0.0')