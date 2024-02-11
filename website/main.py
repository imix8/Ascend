"""
A single-page Taipy application.

Please refer to https://docs.taipy.io/en/latest/manuals/gui/ for more details.
"""

from PIL import Image
from taipy.gui import Gui, Markdown, notify
from pages.root import *
from models.pose import *
import torch
from ultralytics.engine.results import Results, Boxes, Masks, Probs, Keypoints
import os
import cv2
from models.edge_detection import *



home_md = """<|toggle|theme|>

<page|layout|columns=300px 1fr|
<|sidebar|
### Analyzing your **Climb**{: .color-primary} from a video

<br/>
Video Upload
<|{video_path}|file_selector|on_action=upload_video|extensions=.mp4|label=Upload Climb Here!|>

<br/>
Video Download
<|file_download|label=Download Climb Here!|active={fixed}|>
|>

<|container|
# **ASCEND**{: .color-primary}

Give it a try by uploading a video to witness the intricacies of your climb! You can download the processed video in full quality from the sidebar to view. 🧗🏻
<br/>


### Processing Video 📷 
 <|{in_process}|image|>

|>
|page>
"""


# """
# <|layout|columns=1 1|
# <col1|card text-center|part|render={fixed}|
# ### Processing Video 📷 
# <|{in_process}|image|>
# |col1>

# <col2|card text-center|part|render={fixed}|
# ### Processed Video 🔧 
# <|{processed_video}|video|>
# |col2>

# |layout>
# """
video_path = ""
in_process = ''
fixed = False
home = Markdown(home_md)

# function to overlay image 2 over image 1 with alpha
def overlay_image_alpha(img1, img2, alpha=0.5):
    img1 = img1.astype(float)
    img2 = img2.astype(float)
    weighted_sum = cv2.addWeighted(img1, 1, img2, 1 - alpha, 0)
    return weighted_sum

def plot_circle_at_xy(xy, image_path):
    x, y = xy
    image = cv2.imread(image_path)
    cv2.circle(image, (x, y), 10, (0, 0, 255), -1)
    return image

def draw_square_at_bounding_box(xy, image):
    x, y, w, h = xy
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255), 5)
    return image

def begin_analyze(state):
    results = predict_pose(state.video_path)
    for i, r in enumerate(results):
        im_array = r.plot()
        im = Image.fromarray(im_array[..., ::-1])
        im.save(f'temp-{i}.jpg')
        state.in_process = f'temp-{i}.jpg'
        if os.path.exists(f"temp-{i-1}.jpg"):
            os.remove(f"temp-{i-1}.jpg")
        print(r.path)
    print(results)

def upload_video(state):
    notify(state, 'info', 'Uploading original video...')
    notify(state, 'info', 'Processing climb...')
    print(state.video_path)

    n_colors = 3
    holds_image = get_first_frame(state.video_path)
    holds_image = apply_kmeans_image_tracing(holds_image, n_colors=n_colors)
    holds_image = apply_kmeans_image_tracing(draw_bounding_boxes_and_remove(holds_image),n_colors=n_colors,saturation_scale=1,value_scale=1)
    holds_image_masks = get_masks(holds_image)

    results = predict_pose(state.video_path)
    print(results)
    for i, r in enumerate(results):
        keypoints_coords = r.keypoints.xy
        x_col = keypoints_coords[0][:, :1]
        x_average = x_col[x_col != 0].mean()
        y_col = keypoints_coords[0][:, 1:]
        y_average = y_col[y_col != 0].mean()
        im_array = r.plot()
        im = Image.fromarray(im_array[..., ::-1])
        im.save(f'temp-{i}.jpg')

        mask_num = 1
        if not (np.isnan(x_average) or np.isnan(y_average)):
            com = (int(x_average.item()), int(y_average.item()))
            img = plot_circle_at_xy(com, f'temp-{i}.jpg')
            bounding_boxes = find_bounding_boxes_from_mask(holds_image_masks[mask_num])
            for x in x_col:
                for y in y_col:
                    if not (np.isnan(x) or np.isnan(y)):
                        com = (int(x.item()), int(y.item()))
                        for box in bounding_boxes:
                            if com[0] > box[0] and com[0] < box[0] + box[2] and com[1] > box[1] and com[1] < box[1] + box[3]:
                                img = draw_square_at_bounding_box(box, img)
            #cv2.imwrite(f'temp-{i}.jpg', overlay_image_alpha(img, holds_image_masks[mask_num], alpha=0.5))
            cv2.imwrite(f'temp-{i}.jpg', img)

        state.in_process = f'temp-{i}.jpg'
        if os.path.exists(f"temp-{i-1}.jpg"):
            os.remove(f"temp-{i-1}.jpg")

if __name__ == "__main__":
    pages = {
        "/": root_page,
        "home": home,
    }
    gui = Gui(pages=pages)
    gui.md = ""
    gui.run(title="Ascend", use_reloader=True, upload_folder="uploads/", port=8000)
