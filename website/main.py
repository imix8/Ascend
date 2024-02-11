"""
A single-page Taipy application.

Please refer to https://docs.taipy.io/en/latest/manuals/gui/ for more details.
"""

from PIL import Image
from taipy.gui import Gui, Markdown
from pages.root import *
from models.pose import *
import os

home_md = """
# Welcome!

Upload a video of yourself climbing: <|{video_path}|file_selector|label=Select video|extensions=.mp4|>

Upload a photo of the climbing wall without anyone on it: <|{photo_path}|file_selector|label=Select photo|extensions=.jpg,.png|>

How many different colored holds are there: <|{holds}|>


<|{holds}|slider|min=1|max=5|>

<|Begin!|button|on_action=begin_analyze|>

<|{temp}|image|>
"""

video_path = ""
photo_path = ""
holds = 1
temp = ''
home = Markdown(home_md)

def begin_analyze(state):
    results = predict_pose(state.video_path)
    for i, r in enumerate(results):
        im_array = r.plot()
        im = Image.fromarray(im_array[..., ::-1])
        im.save(f'temp-{i}.jpg')
        state.temp = f'temp-{i}.jpg'
        if os.path.exists(f"temp-{i-1}.jpg"):
            os.remove(f"temp-{i-1}.jpg")
        print(r.path)
    print(results)

if __name__ == "__main__":
    pages = {
        "/": root_page,
        "home": home,
    }
    gui = Gui(pages=pages)
    gui.md = ""
    gui.run(title="Ascend", use_reloader=True, upload_folder="uploads/")
