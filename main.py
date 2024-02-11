"""
A single-page Taipy application.

Please refer to https://docs.taipy.io/en/latest/manuals/gui/ for more details.
"""

from PIL import Image
from taipy.gui import Gui, Markdown, notify
from pages.root import *
from models.pose import *
import zipfile
import numpy as np
import os, shutil
import cv2
from models.edge_detection import *


# Markdown template for the home page

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
<|{export_path}|file_download|label=Download Climb Here!|active={fixed}|on_action=download_package|>
|>

<|container|
# **DATA**{: .color-primary}

Give it a try by uploading a video to witness the intricacies of your climb! You can download the processed video in full quality from the sidebar to view. 🧗🏻
<br/>

### Processing Video 📷 
<|{in_process}|image|>

### Center of Mass ⚖️
<|{com_img}|image|>

<|{com_path_img}|image|>

### Utilized Holds 🤙 
 <|{out_utilized_holds}|image|>

|>

|page>
"""

video_path = ""
export_path = ""
in_process = ''
com_img = ''
com_path_img = ''
out_utilized_holds = '' 
fixed = False
home = Markdown(home_md)

# function to overlay image 2 over image 1 with alpha
def overlay_image_alpha(img1, img2, alpha=0.5):
    img1 = img1.astype(float)
    img2 = img2.astype(float)
    weighted_sum = cv2.addWeighted(img1, 1, img2, 1 - alpha, 0)
    return weighted_sum

# function that takes in an image array and an xy coordinate to plot a circle at that point
def plot_circle_at_xy(xy, image):
    x, y = xy
    cv2.circle(image, (x, y), 10, (0, 0, 255), -1)
    return image

# function that takes in an image array and a color and draws a line between two coordinates
def plot_line_between_xy(xy1, xy2, image, color):
    cv2.line(image, xy1, xy2, color, thickness=5)
    return image

# function that takes in an image array and a color and draws a box using the specified coordinates and size
def draw_square_at_bounding_box(xy, image, color):
    x, y, w, h = xy
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 5)
    return image

"""
function to process uploaded video.
processing involves the following steps:
1. Run YOLOv8 Pose Task on the video
2. Calculate the average Keypoint positions for every frame and update frame (CoM)
3. Identify the potential holds for route
4. Find line between feet keypoints and draw line from CoM to line
5. Keep track of holds that were visited

Once all of that has been processed and annotated, the files are saved to specific directories then processed into a video
The videos and pictures are then saved to an archive to download, if the user wishes
"""
def upload_video(state):

    # setting up for processing
    notify(state, 'info', 'Uploading original video...')
    notify(state, 'info', 'Processing climb...')
    if os.path.exists("saves"):
        shutil.rmtree("saves")
    os.mkdir("saves")
    n_colors = 3
    holds_image = get_first_frame(state.video_path)
    cv2.imwrite('saves/base.jpg', holds_image)
    base = cv2.imread("saves/base.jpg")
    cv2.imwrite('saves/com_path.jpg', holds_image)
    holds_image = apply_kmeans_image_tracing(holds_image, n_colors=n_colors)
    holds_image = apply_kmeans_image_tracing(draw_bounding_boxes_and_remove(holds_image),n_colors=n_colors,saturation_scale=1,value_scale=1)
    holds_image_masks = get_masks(holds_image)
    processed_boxes = []

    # runs the pose model on uploaded video 
    results = predict_pose(state.video_path)
    for i, r in enumerate(results):

        # extracting necessary keypoint coordinates
        keypoints_coords = r.keypoints.xy
        x_col = keypoints_coords[0][:, :1]
        non_zero_x = x_col[x_col != 0]
        x_average = non_zero_x.mean()
        y_col = keypoints_coords[0][:, 1:]
        non_zero_y = y_col[y_col != 0]
        
        y_average = non_zero_y.mean()
        im_array = r.plot()
        im = Image.fromarray(im_array[..., ::-1])
        im.save(f'saves/holds-{i}.jpg')
        holds = cv2.imread(f'saves/holds-{i}.jpg')

        im_array = r.plot(img=base, boxes=False)
        im = Image.fromarray(im_array[..., ::-1])
        im.save(f'saves/balance-{i}.jpg')
        balance = cv2.imread(f'saves/balance-{i}.jpg')

        # this check ensures that the full body is present and the x and y coordinates are the same size
        if (non_zero_y.size(dim=0) > 1) and (non_zero_x.size(dim=0) == non_zero_y.size(dim=0)):
            maximum = np.argpartition(non_zero_y, -2)[-2:]
            # draw foot line
            xy1 = (int(non_zero_x[maximum][0]), int(non_zero_y[maximum][0]))
            xy2 = (int(non_zero_x[maximum][1]), int(non_zero_y[maximum][1]))
            balance = plot_line_between_xy(xy1, xy2, balance, (128, 128, 128))

        # getting bounding boxes for all the holds and trying to calculate if the climber is near a hold
        # this part is also is in charge of adding the average point to the images itself
        mask_num = 1
        bounding_boxes = find_bounding_boxes_from_mask(holds_image_masks[mask_num])
        if not (np.isnan(x_average) or np.isnan(y_average)):
            com = (int(x_average.item()), int(y_average.item()))
            com_path = cv2.imread('saves/com_path.jpg')
            cv2.imwrite('saves/com_path.jpg', plot_circle_at_xy(com, com_path))
            holds = plot_circle_at_xy(com, holds)
            balance = plot_circle_at_xy(com, balance)

            # draw line from com to foot line
            if (non_zero_y.size(dim=0) > 1) and (non_zero_x.size(dim=0) == non_zero_y.size(dim=0)):
                xy2 = (int(x_average.item()), int(non_zero_y[maximum][0]))
                x_max = np.sort(non_zero_x[maximum])
                color = (0, 255, 0) if com[0] > x_max[0] and com[0] < x_max[1] else (0, 0, 255)
                balance = plot_line_between_xy(com, xy2, balance, color)
            
            for x in x_col:
                for y in y_col:
                    if not (np.isnan(x) or np.isnan(y)):
                        com = (int(x.item()), int(y.item()))
                        for box in bounding_boxes:
                            if com[0] > box[0] and com[0] < box[0] + box[2] and com[1] > box[1] and com[1] < box[1] + box[3]:
                                holds = draw_square_at_bounding_box(box, holds, (255, 255, 255))
                                if box not in processed_boxes:
                                    processed_boxes.append(box)
            cv2.imwrite(f'saves/holds-{i}.jpg', holds)
        
        cv2.imwrite(f'saves/balance-{i}.jpg', balance)

        state.in_process = f'saves/holds-{i}.jpg'
        state.com_img = f'saves/balance-{i}.jpg'

    # this generates the final image     
    all_boxes = find_bounding_boxes_from_mask(holds_image_masks[1])
    for box in all_boxes:
        if box not in processed_boxes:
            base = draw_square_at_bounding_box(box, base, (0, 0, 255))
        else:
            base = draw_square_at_bounding_box(box, base, (0, 255, 0))
    cv2.imwrite("saves/utilized_holds.jpg", base)

    state.out_utilized_holds = "saves/utilized_holds.jpg"
    state.com_path_img = "saves/com_path.jpg"
    state.fixed = True

    if os.path.exists("exports"):
        shutil.rmtree("exports")
    os.mkdir("exports")

    # preparing archive for download
    balance_imgs = sorted([img for img in sorted(os.listdir("saves"), key=len) if img[0:7] == "balance" and img.endswith(".jpg")], key=len)
    holds_imgs = sorted([img for img in os.listdir("saves") if img[0:5] == "holds" and img.endswith(".jpg")], key=len)  

    generate_video("saves", "balance.mp4", balance_imgs)
    generate_video("saves", "holds.mp4", holds_imgs)
    shutil.copyfile('saves/com_path.jpg', 'exports/com_path.jpg')
    shutil.copyfile('saves/utilized_holds.jpg', 'exports/utilized_holds.jpg')
    shutil.make_archive('exports', 'zip', 'exports')
    state.export_path = 'exports.zip'

# helper function to create video from list of image names
def generate_video(path, name, images):
    frame = cv2.imread(os.path.join(path, images[0]))
    height, width, layers = frame.shape
    
    video = cv2.VideoWriter(f"exports/{name}", 0, 30, (width,height))

    for image in images:
        video.write(cv2.imread(os.path.join(path, image)))
    
    cv2.destroyAllWindows()
    video.release()

# cleans up directories after archive has been deleted
def download_package(state):
    shutil.rmtree("saves")
    shutil.rmtree("exports")
    os.remove("exports.zip")
    state.fixed = False


if __name__ == "__main__":
    pages = {
        "/": root_page,
        "home": home,
    }
    gui = Gui(pages=pages)
    gui.md = ""
    gui.run(title="Ascend", use_reloader=True, upload_folder="uploads/", host='0.0.0.0', port=80, debug=False)