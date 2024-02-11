# Welcome!

Upload a video of yourself climbing: <|{video_path}|file_selector|label=Select video|extensions=.mp4|>

Upload a photo of the climbing wall without anyone on it: <|{photo_path}|file_selector|label=Select photo|extensions=.jpg,.png|>

How many different colored holds are there: <|{holds}|>


<|{holds}|slider|min=1|max=5|>

<|Begin!|button|on_action=begin_analyze|>