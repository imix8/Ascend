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
    results = predict_pose(state.video_path)
    print(results)
    for i, r in enumerate(results):
        keypoints_coords = r.keypoints.xy
        x_col = keypoints_coords[0][:, :1]
        x_average = x_col[x_col != 0].mean()
        y_col = keypoints_coords[0][:, 1:]
        y_average = y_col[y_col != 0].mean()
        com = (x_average.item(), y_average.item())
        im_array = r.plot()
        im = Image.fromarray(im_array[..., ::-1])
        im.save(f'temp-{i}.jpg')

        # Call
        add_avg(com, f'temp-{i}.jpg')

        state.in_process = f'temp-{i}.jpg'
        if os.path.exists(f"temp-{i-1}.jpg"):
            os.remove(f"temp-{i-1}.jpg")
        # dict[str(i)] = json.loads(r.tojson())
    # for key in dict:
    #     for item in dict[key]:
    #         keypoints = item['keypoints']
    #         x_coords = keypoints['x']
    #         y_coords = keypoints['y']
    #         x_mean = mean_exclude_zeros(x_coords)
    #         y_mean = mean_exclude_zeros(y_coords)
    #         keypoints['mean'] = {'x': x_mean, 'y': y_mean}
    # with open(f'{path}/{file_name.split(".")[0]}_processed.json', 'w') as file:
    #     json.dump(dict, file)

    # path_download = f'./post_data/data_{data_id}_processed.mp4'
    # create_data_files.process_video(path_upload)
    
    # path_download = processed_video
    # notify(state, 'success', 'Climb processed successfully!')
    # state.fixed = True
    # state.path_upload = path_upload
    # state.path_download = path_download

if __name__ == "__main__":
    pages = {
        "/": root_page,
        "home": home,
    }
    gui = Gui(pages=pages)
    gui.md = ""
    gui.run(title="Ascend", use_reloader=True, upload_folder="uploads/")
