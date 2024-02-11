import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors


# Edge detect all frames in a video and show a black and white representation of the edges
def edge_detection_video(video_path):
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        cv2.imshow("Edges", edges)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


# Increase the contrast of the image
def increase_contrast(image_path):
    image = cv2.imread(image_path)
    alpha = 1.5
    beta = 0
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    cv2.imshow("Adjusted", adjusted)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Plot scatter of image colors
def plot_colors(image_path):
    image = cv2.imread(image_path)
    r, g, b = cv2.split(image)
    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1, projection="3d")
    pixel_colors = image.reshape((np.shape(image)[0]*np.shape(image)[1], 3))
    norm = colors.Normalize(vmin=-1.0, vmax=1.0)
    norm.autoscale(pixel_colors)
    pixel_colors = norm(pixel_colors).tolist()
    axis.scatter(r.flatten(), g.flatten(), b.flatten(),
                 facecolors=pixel_colors, marker=".")
    axis.set_xlabel("Red")
    axis.set_ylabel("Green")
    axis.set_zlabel("Blue")
    plt.show()

# Do the same thing as show masks, but instead loop over the bounding boxes created in draw_bounding_boxes
# output both the mask of the color and the actual bounding boxes for that color
def show_masks_and_bounding_boxes(image_path, n_colors=3, min_width=10, min_height=10):
    image = get_first_frame(image_path)
    image = apply_kmeans_image_tracing(image, n_colors=n_colors, saturation_scale=2, value_scale=2)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w >= min_width and h >= min_height:
            bounding_boxes.append((x, y, w, h))
    # now lets use the most common color in each bounding box and make the whole bounding box that color
    for x, y, w, h in bounding_boxes:
        box = image[y:y+h, x:x+w]
        box = box.reshape((-1, 3))
        kmeans = KMeans(n_clusters=1).fit(box)
        color = kmeans.cluster_centers_[0]
        image[y:y+h, x:x+w] = color
    # make anything not in a bounding box transparent
    mask = np.zeros_like(edges)
    for x, y, w, h in bounding_boxes:
        mask[y:y+h, x:x+w] = 255
    image = cv2.bitwise_and(image, image, mask=mask)
    image = apply_kmeans_image_tracing(image, n_colors=n_colors, saturation_scale=1, value_scale=1)
    # show each color mask
    colors = all_colors_in_image(image)
    dict_of_boundings = {}
    for x, y, w, h in bounding_boxes:
        for color in colors:
            if color.all() != 0:
                # check if the color is in the bounding box
                # if it is in the bounding box, save it to the dict under the color
                # the same wagy we previously saved the bounding boxes and then
                # we can show the masks for each color in the bounding box
                mask = cv2.inRange(image, color, color)
                mask = cv2.bitwise_not(mask)
                mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                mask = cv2.bitwise_and(image, mask)
                str_color = str(color)
                if np.any(mask[y:y+h, x:x+w]):
                    if str_color in dict_of_boundings:
                        dict_of_boundings[str_color].append((x, y, w, h))
                    else:
                        dict_of_boundings[str_color] = [(x, y, w, h)]
    for color in dict_of_boundings:
        for x, y, w, h in dict_of_boundings[color]:
            mask = cv2.inRange(image, color, color)
            mask = cv2.bitwise_not(mask)
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            mask = cv2.bitwise_and(image, mask)
            cv2.imshow(f"Color {str(color)}", mask)

    cv2.imshow("Bounding Boxes", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
