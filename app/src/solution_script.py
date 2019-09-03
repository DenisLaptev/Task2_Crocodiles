import numpy as np
from cv2 import imread
from os import walk
import matplotlib.pyplot as plt

import cv2

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SIZE = 2
FONT_COLOR = (255, 0, 0)
FONT_THICKNESS = 8

PATH_TO_FOLDER = r'..\resources\crocodiles_imgs'
RESIZE_PERCENT = 50


def resize_image(initial_image, resize_percent):
    new_width = int(initial_image.shape[1] * resize_percent / 100)
    new_height = int(initial_image.shape[0] * resize_percent / 100)
    new_dim = (new_width, new_height)

    # resize image
    resized_image = cv2.resize(initial_image, new_dim, interpolation=cv2.INTER_AREA)

    # print('Original Dimensions : ', initial_image.shape)
    # print('Resized Dimensions : ', resized_image.shape)

    return resized_image


def get_images_list(path_to_folder):
    images_list = []
    for root, dirs, files in walk(path_to_folder):
        for _file in files:
            path_to_file = path_to_folder + '\\' + str(_file)
            image = imread(path_to_file)
            images_list.append(image)
    return images_list


def make_resized_images_list(initial_images_list, resize_percent):
    resized_images_list = []
    for initial_image in initial_images_list:
        resized_image = resize_image(initial_image=initial_image, resize_percent=resize_percent)
        resized_images_list.append(resized_image)
    return resized_images_list


def show_images_list(images_list, image_title):
    count = 1
    for image in images_list:
        image = cv2.putText(image,
                            'image ' + str(count),
                            (10, 50),
                            FONT,
                            FONT_SIZE,
                            FONT_COLOR,
                            FONT_THICKNESS,
                            cv2.LINE_AA)

        image2_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        ###
        img = cv2.medianBlur(image2_gray, 1)

        ret, th1 = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
        th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                    cv2.THRESH_BINARY, 11, 2)
        th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                    cv2.THRESH_BINARY, 11, 2)

        kernel = np.ones((2, 2), np.uint8)
        closing = cv2.morphologyEx(th3, cv2.MORPH_CLOSE, kernel)
        cv2.imshow('closing', closing)
        kernel = np.ones((1, 1), np.uint8)
        dilation = cv2.dilate(closing, kernel, iterations=1)

        erosion = cv2.erode(closing, kernel, iterations=1)
        cv2.imshow('erosion', erosion)

        contours, hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        new_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 100:
                new_contours.append(cnt)

        print("Number of new contours=" + str(len(new_contours)))
        print(new_contours[0])

        cv2.drawContours(image, new_contours, -1, (0, 255, 0), 1)

        cv2.imshow(image_title, image)
        cv2.imshow('dilation', dilation)
        cv2.imshow('image2_gray', image2_gray)

        titles = ['Original Image', 'Global Thresholding (v = 127)',
                  'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
        images = [img, th1, th2, th3]

        for i in range(4):
            plt.subplot(2, 2, i + 1), plt.imshow(images[i], 'gray')
            plt.title(titles[i])
            plt.xticks([]), plt.yticks([])
        plt.show()
        ###

        count += 1

        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    initial_images_list = get_images_list(PATH_TO_FOLDER)
    resized_images_list = make_resized_images_list(initial_images_list, resize_percent=RESIZE_PERCENT)
    show_images_list(resized_images_list, image_title='resized image')


if __name__ == '__main__':
    main()
