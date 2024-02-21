import matplotlib.pyplot as plt
import numpy as np
import cv2


# compute the histogram of an image
def compute_histogram(image):
    histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
    return histogram.astype(np.uint8)


# compute the cdf of the histogram
def compute_cdf(histogram):
    cdf = histogram.cumsum()
    # normalize the cdf
    normalized_cdf = (cdf - cdf.max()) / (cdf.max() - cdf.min())
    return normalized_cdf


# matching the histogram
def histogram_matching(src_img, src_cdf, dst_cdf):
    # Mapping from source image histogram to the destination histogram
    mapping = np.zeros(256)
    dst_index = 0

    for src_index in range(256):
        while dst_cdf[dst_index] < src_cdf[src_index] and dst_index < 255:
            dst_index += 1
        mapping[src_index] = dst_index

    # Finish mapping and apply them to the source image
    matched_img = mapping[src_img].astype(np.uint8)
    return matched_img


# plot image
def plot(image, image_title):
    plt.imshow(image, cmap="gray")
    plt.title(image_title)
    plt.axis("off")
    plt.show()


# load images using cv2.imread
src_img = cv2.imread("plant.jpg", cv2.IMREAD_GRAYSCALE)
dst_img = cv2.imread("car.jpg", cv2.IMREAD_GRAYSCALE)

# initialize 2 histogram which is src_img and dst_img
src_histogram = compute_histogram(src_img)
dst_histogram = compute_histogram(dst_img)

# normalize histogram by divide histogram by its total pixels
src_height, src_width = src_img.shape[:2]
dst_height, dst_width = dst_img.shape[:2]
src_total_pixel = src_height * src_width
dst_total_pixel = dst_height * dst_width

src_histogram = src_histogram / src_total_pixel
dst_histogram = dst_histogram / dst_total_pixel

# cdf of src_img and dst_img
src_cdf = compute_cdf(src_histogram)
dst_cdf = compute_cdf(dst_histogram)

# initialize the matched image
matched_img = histogram_matching(src_img, src_cdf, dst_cdf)

# plot images
plot(src_img, "Source Image")
plot(dst_img, "Destination Image")
plot(matched_img, "Matched Image")