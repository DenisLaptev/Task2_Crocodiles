import cv2
import numpy as np
from os import walk
import matplotlib.pyplot as plt

PATH_TO_FOLDER = r'../resources/images_initial'

PATH_TO_FOLDER_OUTPUT_IMAGES_MODIFIED = r'../output/images_modified'
PATH_TO_FOLDER_OUTPUT_IMAGES_WITH_ALL_CONTOURS = r'../output/images_with_all_contours'
PATH_TO_FOLDER_OUTPUT_IMAGES_WITH_FILTERED_CONTOURS = r'../output/images_with_filtered_contours'

# ADAPTIVE THRESHOLD parameters
MAX_VALUE_THRESHOLD = 255
BLOCK_SIZE = 51
C = 2  # the greater C the more white the result!!!


def get_images_list(path_to_folder):
    images_list = []
    for root, dirs, files in walk(path_to_folder):
        for _file in files:
            path_to_file = path_to_folder + '/' + str(_file)
            image = cv2.imread(path_to_file)
            images_list.append(image)
    return images_list


def show_images_list(images_list):
    count = 1
    for image in images_list:
        image_title = 'image ' + str(count)

        cv2.namedWindow(image_title, cv2.WINDOW_NORMAL)
        cv2.imshow(image_title, image)

        count += 1

        cv2.waitKey(0)
        cv2.destroyAllWindows()


def get_filtered_contours_list(contours):
    new_contours = []
    for cnt in contours:
        # area
        area = cv2.contourArea(cnt)

        # perimeter
        perimeter = cv2.arcLength(cnt, True)

        # area/perimeter <= perimeter /(4*Pi) (=for circle)
        # we asume that area/perimeter > perimeter/ 18

        x, y, w, h = cv2.boundingRect(cnt)

        # aspect ratio = w/h
        aspect_ratio = float(w) / h

        # solidity =  (contour area) / (convex hull area)
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            solidity = float(area) / (hull_area + 0.000001)
        else:
            solidity = float(area) / hull_area

        if ((area > 300 and area < 20000) and (area / perimeter) > (perimeter / 1000)) \
                or ((area > 50 and area < 500) and (area / perimeter) > (perimeter / 30)):
            new_contours.append(cnt)

    return new_contours


def modify_image(image_initial):
    # cvt to gray
    image_gray = cv2.cvtColor(image_initial, cv2.COLOR_BGR2GRAY)

    # Adaptive threshold
    image_ad_thresh = cv2.adaptiveThreshold(src=image_gray,
                                            maxValue=MAX_VALUE_THRESHOLD,
                                            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            thresholdType=cv2.THRESH_BINARY,
                                            blockSize=BLOCK_SIZE,
                                            C=C)

    return image_ad_thresh


def draw_bounding_rectangle_for_contour(contour, image):
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 4)


def get_cells_info(initial_images_list):
    list_of_image_dict = []

    count = 1
    for i in range(len(initial_images_list)):

        image_initial = initial_images_list[i]
        image_initial_copy = image_initial.copy()
        image_modified = modify_image(image_initial)

        file_name = 'image' + str(count) + '.png'

        image_dict = {}

        cells_areas = []
        cells_centers = []
        cells_contours = []

        contours, hierarchy = cv2.findContours(image_modified, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for cnt in contours:
            draw_bounding_rectangle_for_contour(contour=cnt, image=image_initial)

        new_contours = get_filtered_contours_list(contours)
        for cnt in new_contours:

            # area
            area = cv2.contourArea(cnt)

            cells_contours.append(cnt)
            cells_areas.append(area)

            M = cv2.moments(cnt)
            if M['m00'] == 0:
                cx = int(M['m10'] / (M['m00'] + 0.000001))
                cy = int(M['m01'] / (M['m00'] + 0.000001))
            else:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
            cnt_center = (cx, cy)
            cells_centers.append(cnt_center)

        cv2.drawContours(image_initial, contours, -1, (0, 255, 0), 2)
        cv2.drawContours(image_initial_copy, new_contours, -1, (255, 0, 0), 2)

        cv2.namedWindow("all_contours", cv2.WINDOW_NORMAL)
        cv2.imshow("all_contours", image_initial)

        cv2.namedWindow("filtered_contours", cv2.WINDOW_NORMAL)
        cv2.imshow("filtered_contours", image_initial_copy)

        # saving outputs

        path_to_file_output_images_modified = PATH_TO_FOLDER_OUTPUT_IMAGES_MODIFIED + '/' + file_name
        cv2.imwrite(path_to_file_output_images_modified, image_modified)

        path_to_file_output_images_with_all_contours = PATH_TO_FOLDER_OUTPUT_IMAGES_WITH_ALL_CONTOURS + '/' + file_name
        cv2.imwrite(path_to_file_output_images_with_all_contours, image_initial)

        path_to_file_output_images_with_filtered_contours = PATH_TO_FOLDER_OUTPUT_IMAGES_WITH_FILTERED_CONTOURS + '/' + file_name
        cv2.imwrite(path_to_file_output_images_with_filtered_contours, image_initial_copy)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

        image_dict['image'] = image_initial
        image_dict['number of contours'] = len(cells_contours)
        image_dict['cells areas'] = cells_areas
        image_dict['cells centers'] = cells_centers
        image_dict['cells contours'] = cells_contours

        list_of_image_dict.append(image_dict)
        count += 1

    return list_of_image_dict


def main():
    # get list of imeges
    initial_images_list = get_images_list(PATH_TO_FOLDER)

    # show list of images
    show_images_list(initial_images_list)

    # get info about crocodiles cells
    # data structure is a list of dicts.
    # Each dict corresponds to certain image of croco
    # with keys=['image','number of contours','cells areas','cells centers','cells contours']
    list_of_image_dict = get_cells_info(initial_images_list)

    # print list_of_image_dict
    print("----------------contours info----------------")
    for my_dict in list_of_image_dict:
        print(my_dict)
        print("-------------------------")
        print()


if __name__ == '__main__':
    main()
