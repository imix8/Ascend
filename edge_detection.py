import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Edge detect the image and show a black and white representation of the edges
def edge_detection(image, image_path=None):
    if image_path is not None:
        image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return edges
    #cv2.imshow("Edges", edges)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()


# draw bounding boxes around the detected edges from edge_detection
def draw_bounding_boxes(image, image_path=None, min_width=10, min_height=10):
    if image_path is not None:
        image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w >= min_width and h >= min_height:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return image

# draw bounding boxes around the detected edges from edge_detection and remove anything not in a bounding box
# lets also take the corner of each bounding box and see what color it is and then remove that color and annything similar from the image
def draw_bounding_boxes_and_remove(image, image_path=None, min_width=10, min_height=10):
    if image_path is not None:
        image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w >= min_width and h >= min_height:
            bounding_boxes.append((x, y, w, h))
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 0), 2)
    mask = np.zeros_like(edges)
    for x, y, w, h in bounding_boxes:
        mask[y:y+h, x:x+w] = 255
    image = cv2.bitwise_and(image, image, mask=mask)
    # now lets use the most common color in each bounding box and make the whole bounding box that color
    for x, y, w, h in bounding_boxes:
        box = image[y:y+h, x:x+w]
        box = box.reshape((-1, 3))
        kmeans = KMeans(n_clusters=1).fit(box)
        color = kmeans.cluster_centers_[0]
        image[y:y+h, x:x+w] = color
    return image

def apply_kmeans_image_tracing(image, n_colors=8, image_path=None, saturation_scale=2, value_scale=2):
    if image_path is not None:
        image = cv2.imread(image_path)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_image)
    s = cv2.multiply(s, np.array([saturation_scale]))
    v = cv2.multiply(v, np.array([value_scale]))
    s = np.clip(s, 0, 255)
    v = np.clip(v, 0, 255)
    hsv_image = cv2.merge([h, s, v])
    enhanced_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
    pixel_values = enhanced_image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixel_values, n_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    result_image = centers[labels.flatten()]
    result_image = result_image.reshape(enhanced_image.shape)
    return cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)

# doesnt work
def remove_shadows(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    final = cv2.cvtColor(limg, cv2.COLOR_Lab2BGR)
    return final

def print_all_colors_in_image(image, image_path=None):
    if image_path is not None:
        image = cv2.imread(image_path)
    r, g, b = cv2.split(image)
    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1, projection="3d")
    pixel_colors = image.reshape((np.shape(image)[0]*np.shape(image)[1], 3))
    norm = matplotlib.colors.Normalize(vmin=-1.0, vmax=1.0)
    norm.autoscale(pixel_colors)
    pixel_colors = norm(pixel_colors).tolist()
    axis.scatter(r.flatten(), g.flatten(), b.flatten(),
                 facecolors=pixel_colors, marker=".")
    axis.set_xlabel("Red")
    axis.set_ylabel("Green")
    axis.set_zlabel("Blue")
    plt.show()

if __name__ == "__main__":
    n_colors = 5
    image = cv2.imread("assets/climbing_wall.jpg")
    cv2.imshow(f"Original", image)
    image = apply_kmeans_image_tracing(image, n_colors=n_colors)
    cv2.imshow(f"Rasterized", image)
    image = draw_bounding_boxes_and_remove(image)
    cv2.imshow(f"Image {n_colors}", image)
    #print_all_colors_in_image(image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
