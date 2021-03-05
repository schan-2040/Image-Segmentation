"""
File: image_segmentor.py
Author : Shaurya Chandhoke
Description: Command line script that takes an image path as input and processes the image as output
"""
import argparse
import collections
import os
import time

import cv2

from src.kmeans import kmeans_process
from src.slic import slic_process


def output_processing(originalImage1, originalImage2, kmeans_image, slic_image, timeElapsed, nosave, noshow):
    """
    The final stages of the program. This function will display the images and/or write them to files as well as
    provide an execution time.

    :param originalImage1: white-tower.png
    :param originalImage2: wt_slic.png
    :param kmeans_image: The K Means clustering algorithm image generated against white-tower.png
    :param slic_image: The SLIC segmentation algorithm image generated against wt_slic.png
    :param timeElapsed: The total runtime for the program
    :param nosave: Flag that determines whether to save the images to the ./out/ directory
    :param noshow: Flag that determines whether to show the images as output
    """
    if (noshow is True) and (nosave is True):
        print("(BOTH FLAGS ON) Recommend disabling either --nosave or --quiet to capture processed images")
        return 0

    print("=" * 40)
    print("Rendering Images...")

    if noshow is False:
        print("(DISPLAY ON) The ESC key will close all pop ups")
        cv2.imshow("white-tower.png", originalImage1)
        cv2.imshow("wt_slic.png", originalImage2)
        cv2.imshow("k-means-white-tower.png", kmeans_image)
        cv2.imshow("slic_wt_slic.png", slic_image)
        cv2.waitKey(0)

    if nosave is False:
        print("(IMAGE SAVE ON) Images are being written to the ./out/ folder")
        cv2.imwrite("./out/step_1_kmeans_white_tower.jpg", kmeans_image)
        cv2.imwrite("./out/step_2_wt_slic.jpg", slic_image)

    print("(DONE): You may want to rerun the program with the --help flag for more options to fine tune the program")
    print("=" * 40)
    print("Time to Process Image: {} seconds.".format(timeElapsed))


def start(prob1Image, prob2Image, k):
    """
    Starter function responsible for beginning the process for obtaining edges

    :param prob1Image: The input image for K-Means segmentation
    :param prob2Image: The input image for SLIC segmentation
    :param k: Cluster number
    :return: A series of copies of the original images passed through the segmentation algorithms
    """
    print("Please wait, processing image and returning output...\n")

    kmeans_image = kmeans_process(prob1Image, k)

    print()

    slic_image = slic_process(prob2Image.copy())

    print("\r\n")

    return kmeans_image, slic_image


def main():
    """
    Beginning entry point into the edge detection program.
    It will first perform prerequisite steps prior to starting the intended program.
    Upon parsing the command line arguments, it will trigger the start function
    """

    '''
    As per the homework assignment, we are to process only the following images:
        white-tower.png
        wt_slic.png
    
    By saving these names into a variable, these two files can be saved in a directory containing multiple other files.
    This program will be able to extract them from the directory if they exist.    
    '''
    imageNames = ['white-tower.png', 'wt_slic.png']

    # Reusable message variables
    ADVICE = "rerun with the (-h, --help) for more information."

    # Start cli argparser
    temp_msg = "Given the path to a directory containing white-tower.png and wt_slic.png, this program will perform " \
               "image segmentation"
    parser = argparse.ArgumentParser(prog="image_segmentor.py", description=temp_msg,
                                     usage="%(prog)s [dirpath] [flags]")

    temp_msg = "The directory path containing the two .png files. The path can be relative or absolute."
    parser.add_argument("dirpath", help=temp_msg, type=str)

    temp_msg = "The number of clusters to create for K-Means Clustering. Default is 10 clusters."
    parser.add_argument("-k", "--kclusters", help=temp_msg, type=int, default=10)

    temp_msg = "If passed, the images will not be written to a file. By default, images are written."
    parser.add_argument("-n", "--nosave", help=temp_msg, action="store_true")

    temp_msg = "If passed, the images will not be displayed. By default, the images will be displayed."
    parser.add_argument("-q", "--quiet", help=temp_msg, action="store_true")

    # Obtain primary CLI arguments
    args = parser.parse_args()
    dirpath = args.dirpath
    kclusters = args.kclusters
    nosave = args.nosave
    noshow = args.quiet

    # Begin error checking params and checking the validity of the directory
    if os.path.exists(dirpath):
        # Obtain all files within the directory and extract only the ones that are to be processed
        files = list(filter(lambda file: file in imageNames, os.listdir(dirpath)))
        if collections.Counter(files) != collections.Counter(imageNames):
            print("Error: Cannot find files within this directory.\nPlease check if the path is correct or " + ADVICE)
            return -1
        else:
            cleanedPath = os.path.abspath(dirpath)
            files = [os.path.join(cleanedPath, file) for file in files]
    else:
        print("Error: Cannot find directory.\nPlease check if the path is correct or " + ADVICE)
        return -1

    if 'white-tower.png' not in files[0]:
        temp = files[0]
        files[0] = files[1]
        files[1] = temp

    problem1_Img = cv2.imread(files[0], cv2.IMREAD_COLOR)
    problem2_Img = cv2.imread(files[1], cv2.IMREAD_COLOR)

    START_TIME = time.time()

    kmeans_image, slic_image = start(problem1_Img, problem2_Img, kclusters)

    ELAPSED_TIME = time.time() - START_TIME

    output_processing(problem1_Img, problem2_Img, kmeans_image, slic_image, ELAPSED_TIME, nosave, noshow)


if __name__ == "__main__":
    main()
