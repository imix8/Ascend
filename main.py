from taipy.gui import Gui, notify
from PIL import Image
from io import BytesIO
import cv2
import create_data_files


path_upload = ""
path_download = ""

original_video = None
processed_video = None
fixed = False
data_id = 0
slider_value = 15

page = """<|toggle|theme|>

<page|layout|columns=300px 1fr|
<|sidebar|
### Analyzing your **Climb**{: .color-primary} from a video

<br/>
Video Upload
<|file_selector|on_action=upload_video|extensions=.mp4|label=Upload Climb Here!|>

<br/>
Video Download
<|file_download|label=Download Climb Here!|active={fixed}|>
|>

<|container|
# **ASCEND**{: .color-primary}

Give it a try by uploading a video to witness the intricacies of your climb! You can download the processed video in full quality from the sidebar to view. 🧗🏻
<br/>

<videos|layout|columns=1 1|
<col1|card text-center|part|render={fixed}|
### Original Video 📷 
<|{original_video}|video|>
|col1>

<col2|card text-center|part|render={fixed}|
### Processed Video 🔧 
<|{processed_video}|video|>
|col2>
|videos>

|>
|page>
"""

# <!-- Slider input added here -->
# <br/>
# Set Number of Hold Colors
# <|slider|min=0|max=100|value={slider_value}|on_change=set_slider_value|label=Adjust Quality:|>
# # |>

# <|container|text-center|
# <jpg src='logo.jpg' alt='Ascend Logo' style='max-width: 100%; height: auto;'>
# |container>

# def set_slider_value(state, value):
#     global slider_value
#     state.slider_value = value
#     notify(state, 'success', 'Number set successfully!')


def upload_video(state, file_info):
    global data_id, path_download, original_video, processed_video, fixed
    data_id += 1

    notify(state, 'info', 'Uploading original video...')
    video_content = file_info['content']
    path_upload = f'data_{data_id}.mp4'
    with open(path_upload, 'wb') as file:
        file.write(video_content)

    # path_download = f'./post_data/data_{data_id}_processed.mp4'
    # notify(state, 'info', 'Processing climb...')
    # create_data_files.process_video(path_upload)
    
    # path_download = processed_video
    # notify(state, 'success', 'Climb processed successfully!')
    # state.fixed = True
    # state.path_upload = path_upload
    # state.path_download = path_download

if __name__ == '__main__':
    Gui(page=page).run(margin="0px", title='ASCEND: Climb Analysis Tool', upload_folder="pre_data/")


# from taipy.gui import Gui, notify
# from rembg import remove
# from PIL import Image
# from io import BytesIO


# path_upload = ""
# path_download = "fixed_img.png"
# original_image = None
# fixed_image = None
# fixed = False


# page = """<|toggle|theme|>

# <page|layout|columns=300px 1fr|
# <|sidebar|
# ### Removing **Background**{: .color-primary} from your image

# <br/>
# Upload and download
# <|{path_upload}|file_selector|on_action=fix_image|extensions=.png,.jpg|label=Upload original image|>

# <br/>
# Download it here
# <|{path_download}|file_download|label=Download fixed image|active={fixed}|>
# |>

# <|container|
# # Image Background **Eliminator**{: .color-primary}

# 🐶 Give it a try by uploading an image to witness the seamless removal of the background. You can download images in full quality from the sidebar.
# This code is open source and accessible on [GitHub](https://github.com/Avaiga/demo-remove-background).
# <br/>


# <images|layout|columns=1 1|
# <col1|card text-center|part|render={fixed}|
# ### Original Image 📷 
# <|{original_image}|image|>
# |col1>

# <col2|card text-center|part|render={fixed}|
# ### Fixed Image 🔧 
# <|{fixed_image}|image|>
# |col2>
# |images>

# |>
# |page>
# """


# def convert_image(img):
#     buf = BytesIO()
#     img.save(buf, format="PNG")
#     byte_im = buf.getvalue()
#     return byte_im


# def fix_image(state):
#     notify(state, 'info', 'Uploading original image...')
#     image = Image.open(state.path_upload)
    
#     notify(state, 'info', 'Removing the background...')
#     fixed_image = remove(image)
#     fixed_image.save("fixed_img.png")

#     notify(state, 'success', 'Background removed successfully!')
#     state.original_image = convert_image(image)
#     state.fixed_image = convert_image(fixed_image)
#     state.fixed = True

# if __name__ == "__main__":
#     Gui(page=page).run(margin="0px", title='Background Remover')

# if __name__ == '__main__':
#     main()