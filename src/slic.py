"""
File: slic.py
Author: Shaurya Chandhoke
Description: Helper file which contains the function used for image segmentation via the SLIC algorithm
"""
import numpy as np
import pandas as pd

from tqdm import tqdm
from scipy.spatial import distance

from src.image_output_processing import logger
from src.gradient import gradient_process


def cartesianCoordinates(image):
    """
    Helper function that converts the original input image into a 2D numpy array of cartesian coordinates.

    :param image: The original input image
    :return: The cartesian array of the image
    """
    nrow = image.shape[0]
    ncol = image.shape[1]
    rowIndex = np.arange(0, nrow)
    colIndex = np.arange(0, ncol)

    rowIndices = np.array([np.repeat(rowIndex, ncol)])
    colIndices = np.array([np.tile(colIndex, nrow)])
    indexMatrix = np.transpose(np.vstack((rowIndices, colIndices)))

    return indexMatrix


def gradientComputation(image):
    """
    Helper function that computes the gradient of the image in each channel and returns the square root of the sum of
    squares of the all 3 channel gradients.

    :param image: The input image
    :return: The gradient of the image
    """
    imageR = np.float64(image[:, :, 2])
    imageG = np.float64(image[:, :, 1])
    imageB = np.float64(image[:, :, 0])

    Gradient_R = gradient_process(imageR)
    Gradient_G = gradient_process(imageG)
    Gradient_B = gradient_process(imageB)

    return np.sqrt(Gradient_R + Gradient_G + Gradient_B)


def initializeCentroids(image, blockSize=50):
    """
    Helper function that will determine the centroid location of the super pixels

    :param image: The input image
    :param blockSize: The initial size of each super pixel
    :return: The initial tracked centroid locations
    """
    centroidCollection = []
    initPoint = int(blockSize / 2)

    for i in range(initPoint, image.shape[0], blockSize):
        for j in range(initPoint, image.shape[1], blockSize):
            centroid = [i, j] + list(image[i, j, :])
            centroidCollection.append(centroid)

    return np.asarray(centroidCollection)


def localShift(gradient, centroidArray, image):
    """
    A core function that will shift a centroid away from an image gradient's edge. This is done to minimize noise
    during the algorithm's cluster assignment process.

    :param gradient: The gradient of the input image
    :param centroidArray: The tracked centroid locations
    :param image: The original input image
    :return: The new centroid locations shifted if needed
    """
    shiftedCentroidArray = []
    for centroid in centroidArray:
        centroid = centroid.astype(np.int)

        windowX = centroid[0] - 1, centroid[0] + 2
        windowY = centroid[1] - 1, centroid[1] + 2

        windowRangeX = np.arange(windowX[0], windowX[1])
        windowRangeY = np.arange(windowY[0], windowY[1])

        window = gradient[windowX[0]:windowX[1], windowY[0]:windowY[1]]
        smallestGradient = np.where(window == np.amin(window))
        localMinima = [smallestGradient[0][0], smallestGradient[1][0]]

        newCentroid = windowRangeX[localMinima[0]], windowRangeY[localMinima[1]]
        centroid[:2] = newCentroid
        centroid[2:] = image[newCentroid[0], newCentroid[1], :]

        shiftedCentroidArray.append(centroid)

    return np.asarray(shiftedCentroidArray)


def closestCluster(image, cartesianArray, centroidCollection):
    """
    A core function that will determine a pixel's closest superpixel using the L2 norm. This function employs a C-based
    function for determining the closest distance using the L2 norm as previous designs of this function yielded a high
    time complexity.

    :param image: The image as an array of x,y,R,G,B vectors
    :param cartesianArray: The image as a vector or cartesian coordinates
    :param centroidCollection: The tracked centroid locations
    :return: The super pixel dataframe
    """
    clusterID_collection = []
    nrows = image.shape[0] * image.shape[1]
    rgbSpace = image.reshape(nrows, 3)

    imageSpace = np.float32(np.concatenate((cartesianArray, rgbSpace), axis=1))

    centroidCollection[:, :2] = centroidCollection[:, :2] / 2

    # Because the process below can take quite some time, a progress bar library is being used to allow the user to
    # view approximately how much longer the wait time will be
    for pixel in tqdm(imageSpace, ncols=100):
        standardPixel = pixel.copy()
        standardPixel[:2] = standardPixel[:2] / 2

        clusterID = distance.cdist(np.array([standardPixel]), centroidCollection).argmin()
        clusterID_collection.append(clusterID)

    clusterID_collection = np.asarray(clusterID_collection)
    clusterFrame = pd.DataFrame(imageSpace, columns=['x', 'y', 'r', 'g', 'b'])
    clusterFrame['cluster'] = clusterID_collection

    return clusterFrame


def aggregateMean(clusterFrame, default_round=4):
    """
    Helper function to aggregate the tracked superpixels by cluster ID and return the mean of the feature space

    :param clusterFrame: The super pixel dataframe
    :param default_round: A default value to round the mean
    :return: The superpixel mean dataframe
    """
    aggregatedFrame = clusterFrame.groupby('cluster').mean().apply(np.round, axis=1, decimals=default_round)

    return aggregatedFrame


def updateCentroids(meanFrame):
    """
    Helper function to convert the aggregated mean feature space to a 2D numpy array

    :param meanFrame: The aggregated mean dataframe
    :return: The dataframe as a 2D numpy array
    """
    return np.asarray(meanFrame)


def paintImage(image, clusterFrame, centroidCollection, originalDimensions):
    """
    Helper function that converts the image back into a 2D numpy array for post processing

    :param image: The image as an array of x,y,R,G,B vectors
    :param clusterFrame: The super pixel dataframe
    :param centroidCollection: The tracked cluster centroids
    :param originalDimensions: The original dimensions of the image
    :return: The image and redrawn with borders
    """
    clusterID = clusterFrame.iloc[:, -1]

    nrows = image.shape[0] * image.shape[1]
    image = image.reshape(nrows, 3)

    for i in range(image.shape[0]):
        clusterid = clusterID[i]
        rgbVector = centroidCollection[clusterid, 2:]
        image[i] = rgbVector

    finalizedImage = image.reshape(originalDimensions)
    finalizedImage = drawBorders(finalizedImage)

    return finalizedImage


def drawBorders(image):
    """
    Helper function to draw borders around pixels of differentiating intensity, indicating the pixel is a border

    :param image: The superpixeled image as a 2D numpy array
    :return: The image with borders drawn
    """
    for i in range(image.shape[0]):
        for j in range(1, image.shape[1] - 1):
            pixel = image[i, j, :]
            previousPixel = image[i, j - 1, :]
            nextPixel = image[i, j + 1, :]

            if (not np.array_equal(pixel, previousPixel)) and (np.array_equal(pixel, nextPixel)):
                image[i, j - 1, :] = 0

    for j in range(image.shape[1]):
        for i in range(1, image.shape[0] - 1):
            pixel = image[i, j, :]
            previousPixel = image[i - 1, j, :]
            nextPixel = image[i + 1, j, :]

            if (not np.array_equal(pixel, previousPixel)) and (np.array_equal(pixel, nextPixel)):
                image[i, j - 1, :] = 0

    return image


def slic_process(image):
    """
    Entry point into the SLIC Segmentation algorithm for image segmentation. Given an image, this algorithm will try to
    cluster pixels into larger superpixels.

    NOTE: This algorithm is naively designed
        The biggest challenge was trying to minimize the cost of determining the closest cluster per pixel via the
        L2 norm. Given that the wt_slice.png was 500x750 = 375,000 pixels and that the blocksize was 50x50 = 2,500
        pixels, the number of superpixels to create was 375,000/2,500 = 150. This meant that the cost to determine
        the closest centroid was approximately 375,000*150 = 56,250,000 cycles per iteration.

        Knowing Python's underlying bottleneck for interpretability vs speed, a special C based function was employed to
        obtain the nearest centroid using the L2 Norm. This cut down the computation cost by roughly 7-10 minutes PER
        ITERATION. Originally, the program would take 7-10 minutes per iteration, whereas it will now take about 6
        seconds.

    :param image: The original image as a 2D numpy array
    :return: The image repainted with borders around the superpixels
    """
    logger('2', 0, 'Performing SLIC segmentation')

    print()

    centroidCollection = initializeCentroids(image)
    cartesianArray = cartesianCoordinates(image)
    gradient = gradientComputation(image)

    clusterFrame = None
    for i in range(3):
        centroidCollection = localShift(gradient, centroidCollection, image)
        clusterFrame = closestCluster(image, cartesianArray, centroidCollection.copy())

        meanFrame = aggregateMean(clusterFrame)
        centroidCollection = updateCentroids(meanFrame)

    logger('2a', 0, 'Drawing borders', substep=True)
    finalizedImage = paintImage(image, clusterFrame, centroidCollection, image.shape)
    logger('2a', 1, 'Drawing borders', substep=True)

    logger('2', 1, 'Performing SLIC segmentation')

    return finalizedImage
