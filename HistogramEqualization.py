import matplotlib.pyplot as plt
import numpy as np
import cv2


# compute the histogram of an image
def compute_histogram(image):
    histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
    return histogram.astype(np.uint8)


# equalized the histogram
def histogram_equalization(origin_histogram):
    # calculate the histogram cumulative
    cdf = origin_histogram.cumsum()
    cdf_normalized = (cdf - cdf.min()) / (cdf.max() - cdf.min())
    return cdf_normalized


# plot the image and the histogram
def plot(histogram, image, title):
    fig = plt.figure(figsize=(10, 5))
    ax = plt.subplot(1, 2, 1)
    plt.imshow(image, cmap="gray")
    plt.title(title)
    plt.axis("off")

    plt.subplot(122)
    plt.plot(histogram)
    plt.show()


# load image
image = cv2.imread("car.jpg", cv2.IMREAD_GRAYSCALE)

# calculate the image's histogram and plot it
origin_histogram = compute_histogram(image)
plot(origin_histogram, image, "Origin")

# calculate the new histogram of an image and set the new histogram to the image
equalized_histogram = histogram_equalization(origin_histogram)
equalized_image = np.interp(image, np.arange(0, 256), equalized_histogram)
plot(equalized_histogram, equalized_image, "Equalized")