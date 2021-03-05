"""
File: kmeans.py
Author: Shaurya Chandhoke
Description: Helper file which contains the function used for image segmentation via the K-Means algorithm
"""
import pandas as pd
import numpy as np

from scipy.spatial import distance
from tqdm import tqdm
from src.image_output_processing import logger


def initSeed(image, k):
    """
    A helper function to randomly instantiate the beginning centroid locations, as well as reshaping the original image
    for the rest of the algorithm.

    :param image: The original image to reshape (from (row x column x 3) to (row * column) x k space)
    :param k: The number of clusters. Also determines the reshaping of the image.
    :return: The random centroid locations as well as the reshaped image
    """
    nrows = image.shape[0] * image.shape[1]
    image = image.reshape(nrows, 3)

    seedIndex = np.random.randint(0, nrows, size=k)
    seed = image[seedIndex, :]

    return seed, image.copy()


def euclidean_distance(centroid, vector):
    """
    A helper function that makes use of numpy's norm functions to calculate the l2 norm.

    :param centroid: The centroid reference array to determine which cluster the rgb vector is closest to
    :param vector: A pixel rgb vector pulled from the image
    :return: The l2 norm
    """
    return np.linalg.norm(centroid - vector)


def aggregateMeans(clusterFrame, default_round=4):
    """
    A helper function that will group the dataframe of rgb vectors and respective cluster ids and returns the mean per
    cluster.

    :param clusterFrame: Dataframe representing the rgb vectors in the image and their respective cluter id
    :param default_round: Determines how precise the mean should be in calculation
    :return: A dataframe containing each cluster's mean at that iteration
    """
    return clusterFrame.groupby('cluster').mean().apply(np.round, axis=1, decimals=default_round)


def meanDiff(aggregatedMeanFrame, previousMeans):
    """
    Helper function to determine the change in mean values each centroid has obtained from the previous run. This is
    the function to determine if the algorithm has converged or not.

    The convergence is evaluated by the following:
        Assume k is 5 (in other words 5 centroids), and the their respective means at iteration 14 are approximately:
            [22.5, 30.2, 10.6, 89.8, 109.9]

        If at iteration 15, their new means are approximately:
            [22.4, 29.9, 10.6, 89.7, 109.6]

        Then the difference in sums of means is approximately 0.8. Depending on the CONVERGENCE_THRESHOLD, which
        determines the maximum value in the difference of sums of means, this score of 0.8 will be checked to see
        if it constitutes a convergence.


    :param aggregatedMeanFrame: The dataframe containing each cluster's means
    :param previousMeans: The previous iteration's means
    :return: The difference in sums of means
    """
    clusterMeans = np.asarray(aggregatedMeanFrame.sum(axis=1))

    if previousMeans.size == 0:
        return np.sum(clusterMeans)
    else:
        previousMeans = np.asarray(previousMeans.sum(axis=1))
        return np.sum(np.abs(clusterMeans - previousMeans))


def closestCluster(rgbPoints, centroids):
    """
    Helper function to assign a rgb vector to a cluster based on the l2 norm.

    :param rgbPoints: A 2D numpy array representing the input image as an array of rgb vectors
    :param centroids: The centroid reference array to determine which cluster the rgb vector is closest to
    :return: A dataframe containing each rgb vector and it's newly assigned cluster
    """
    closestCentroids = []
    redundancyCheck = centroids.tolist()

    # Because the process below can take quite some time, a progress bar library is being used to allow the user to
    # view approximately how much longer the wait time will be
    rgbPoints = tqdm(rgbPoints, ncols=100)

    for rgbVector in rgbPoints:
        # Only check pixels that are not already centroids
        if rgbVector.tolist() not in redundancyCheck:
            closestIndex = distance.cdist(np.array([rgbVector]), centroids).argmin()
            closestCentroids.append(closestIndex)
        else:
            closestIndex = np.argwhere(rgbVector == centroids)[0, 0]
            closestCentroids.append(closestIndex)

    clusterFrame = pd.DataFrame(rgbPoints, columns=['r', 'g', 'b'])
    clusterFrame['cluster'] = closestCentroids

    return clusterFrame


def paintImage(clusterID, centroids, rgbPoints, originalDimensions):
    """
    Helper image to repaint the image (represented as a 2D matrix of rgb vectors)

    :param clusterID: The finalized id given to each rgb vector in the rgbPoints
    :param centroids: The centroid reference array to repaint the image with
    :param rgbPoints: A 2D numpy array representing the input image as an array of rgb vectors
    :param originalDimensions: The original dimensions of the input image
    :return: The repainted image
    """
    for i in range(rgbPoints.shape[0]):
        clusterid = clusterID[i]
        newRGBVector = centroids[clusterid]
        rgbPoints[i] = newRGBVector

    finalizedImage = rgbPoints.reshape(originalDimensions)
    finalizedImage = np.uint8(finalizedImage)

    return finalizedImage


def kmeans_process(image, k):
    """
    Entry point into the K Means Clustering algorithm for image segmentation. Given the number of clusters, k, this
    algorithm will create k random centroids in the rgb image space and calculate the correctness of the clusters
    generated per centroid.

    Being an unsupervised algorithm, it will continue to perform iterations until all centroids no longer change in mean
    values from previous iterations at which point it is said the centroids have converged to a particular rgb value.

    Once that convergence occurs, each pixel in the image is repainted to its corresponding cluster id and returned.

    NOTE: This algorithm is naively designed and is dependent on the following pieces of information at runtime:
        - Image dimensions
        - Number of clusters
        - Initial centroid locations

        Due to this, the time complexity PER ITERATION is approximately O(kn) where
            - k is the number of clusters
            - n is the number of pixels (row * column)

        The biggest challenge was trying to minimize the cost of determining the closest centroid per pixel via the
        L2 norm. white-tower.png has image dimensions of 1280x720 = 921,600 pixels. Given the number of clusters to
        create, k, the cost to determine the closest centroid is O(k*921,600) per iteration.

        Knowing Python's underlying bottleneck for interpretability vs speed, a special C based function was employed to
        obtain the nearest neighbor using the L2 Norm. This cut down the computation cost by roughly 7-10 minutes PER
        ITERATION. Originally, the program would take 7-10 minutes per iteration, whereas it will now take about 10
        seconds.

    :param image: The original image as a 2D numpy array
    :param k: The number of clusters
    :return: The image repainted with k segmentations
    """
    CONVERGENCE_THRESHOLD = 2
    diff = 1000
    previousDiff = 0
    originalDimensions = image.shape

    logger('1', 0, 'Performing K-Means Segmentation', False, 'k={}'.format(k))

    print("\r\n")

    centroids, rgbPoints = initSeed(image, k)
    previousMeanFrame = pd.DataFrame(np.random.randint(0, 255, size=(k, 3)))

    i = 1

    clusterFrame = None
    while diff >= CONVERGENCE_THRESHOLD:
        clusterFrame = closestCluster(rgbPoints, centroids)
        meanFrame = aggregateMeans(clusterFrame)

        newDiff = meanDiff(meanFrame, previousMeanFrame)
        diff = abs(round(newDiff - previousDiff, ndigits=3))

        print('\r\n[Iteration: {} Complete]\r\nCluster Mean Convergence Diff: {}\r\n'.format(i, diff))
        if diff >= CONVERGENCE_THRESHOLD:
            previousDiff = newDiff
            centroids = np.asarray(meanFrame)
            previousMeanFrame = meanFrame.copy()

        i += 1

    clusterID = np.asarray(clusterFrame.iloc[:, -1])
    finalImage = paintImage(clusterID, centroids, rgbPoints, originalDimensions)

    logger('1', 1, 'K-Means Segmentation Has Converged', False, 'k={}'.format(k))

    return finalImage
