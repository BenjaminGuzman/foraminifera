import os
import sys
import shutil

import cv2
import numpy as np

import progressbar

DATASET_DIR = "../foraminifera-dataset"
IMG_DIR = "img"
OUT_SEGMENTED_IMAGES_DIR = "img/segmented"
OUT_SEGMENTED_CROPPED_IMAGES_DIR = "img/segmented_cropped"
OUT_SPECIES_IMAGES_DIR = "img/species"


def binarize_image(image):
    image = cv2.GaussianBlur(image, (5, 5), 0)  # make smoother contours
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # so we can do binarization below
                                                # empiric threshold
    ret, image_thresh = cv2.threshold(gray_image, 30, 255, cv2.THRESH_BINARY)  # binarization
    return image_thresh


def get_contours(image_bin, original_image=None):
    contours, hierarchy = cv2.findContours(image_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    image_with_contours = None
    if original_image is not None:
        image_with_contours = cv2.drawContours(original_image, contours, -1, (255, 0, 255))

    return contours, image_with_contours


def get_and_clean_roi(image_bin, original_image):
    contours, _ = get_contours(image_bin)

    contour_areas = [(contour, cv2.contourArea(contour)) for contour in contours]

    # obtain the greatest contour in the image, possibly the contour of the foraminifera
    max_contour = max(contour_areas, key=lambda contour_tuple: contour_tuple[1])
    max_contour_contour, max_contour_area = max_contour

    # create a new image with only the ROI
    # this image will be the mask
    image_shape = image_bin.shape[:2]  # width & height, 3rd position is the number of channels
    black_image = np.zeros(image_shape, np.uint8)
    image_mask = cv2.fillConvexPoly(black_image, max_contour_contour, (255, 255, 255))

    # apply mask
    image_roi = cv2.bitwise_and(original_image, original_image, mask=image_mask)

    return image_roi


def clear_images():
    DEBUGGING = False
    MAX_FILES = 10
    i = 0

    progress_bar = progressbar.ProgressBar(max_value=progressbar.UnknownLength, redirect_stdout=True)

    for specie_dir in os.listdir(OUT_SPECIES_IMAGES_DIR):

        specie_dir_path = os.path.join(OUT_SPECIES_IMAGES_DIR, specie_dir)

        specie_dir_out_path = os.path.join(OUT_SEGMENTED_IMAGES_DIR, specie_dir)
        if not os.path.isdir(specie_dir_out_path):
            os.mkdir(specie_dir_out_path)

        for image_name in os.listdir(specie_dir_path):
            image_path = os.path.join(specie_dir_path, image_name)

            if not os.path.isfile(image_path):
                continue

            image = cv2.imread(image_path)

            image_bin = binarize_image(image)

            image_roi = get_and_clean_roi(image_bin, image)

            if DEBUGGING:
                cv2.imshow("Original", image)
                cv2.imshow("Binarized", image_bin)
                cv2.imshow("ROI", image_roi)
                cv2.waitKey(0)

                if i > MAX_FILES:
                    return
            else:
                file_abs_path = os.path.join(specie_dir_out_path, image_name)
                cv2.imwrite(file_abs_path, image_roi)
                progress_bar.update(i)

            i += 1

    return 0


def crop_roi_images():
    progress_bar = progressbar.ProgressBar(max_value=progressbar.UnknownLength, redirect_stdout=True)
    i = 0

    for specie_dir in os.listdir(OUT_SEGMENTED_IMAGES_DIR):

        specie_dir_path = os.path.join(OUT_SEGMENTED_IMAGES_DIR, specie_dir)

        specie_dir_out_path = os.path.join(OUT_SEGMENTED_CROPPED_IMAGES_DIR, specie_dir)
        if not os.path.isdir(specie_dir_out_path):
            os.mkdir(specie_dir_out_path)

        for image_name in os.listdir(specie_dir_path):
            image_path = os.path.join(specie_dir_path, image_name)

            if not os.path.isfile(image_path):
                continue

            image = cv2.imread(image_path)
            image_bin = binarize_image(image)

            contours, _ = get_contours(image_bin)

            # get the ROI, the foraminifera contour
            max_contour = max(contours, key=cv2.contourArea)

            min_point = [
                min(max_contour, key=lambda contour: contour[0][0])[0][0],
                min(max_contour, key=lambda contour: contour[0][1])[0][1]
            ]
            max_point = [
                max(max_contour, key=lambda contour: contour[0][0])[0][0],
                max(max_contour, key=lambda contour: contour[0][1])[0][1]
            ]

            # crop with a square the image
            image_cropped = image[min_point[1]:max_point[1], min_point[0]:max_point[0]]

            file_abs_path = os.path.join(specie_dir_out_path, image_name)

            cv2.imwrite(file_abs_path, image_cropped)
            progress_bar.update(i)

            i += 1

    return 0


def check_dirs():
    if not os.path.isdir(DATASET_DIR):
        sys.stderr.write("\"{}\" DOES NOT EXISTS, exiting...".format(DATASET_DIR))
        sys.exit(1)

    for dir_path in [IMG_DIR, OUT_SEGMENTED_IMAGES_DIR, OUT_SPECIES_IMAGES_DIR, OUT_SEGMENTED_CROPPED_IMAGES_DIR]:
        if not os.path.isdir(dir_path):
            print("Directory \"{}\" does not exists, creating directory...".format(dir_path))
            os.mkdir(dir_path)


def move_images():
    progress_bar = progressbar.ProgressBar(max_value=progressbar.UnknownLength, redirect_stdout=True)
    species_count = {}

    # separate foraminifera species
    blacklist_dirs = ["Others", "ReadMe.txt"]
    for specie_name in os.listdir(DATASET_DIR):
        if specie_name in blacklist_dirs:
            continue

        species_count[specie_name] = 0

        specie_directory_abs_path = os.path.join(OUT_SPECIES_IMAGES_DIR, specie_name)
        directory = os.path.join(DATASET_DIR, specie_name)

        if not os.path.isdir(specie_directory_abs_path):
            os.mkdir(specie_directory_abs_path)

        if os.path.isdir(directory):
            for sub_directory in os.listdir(directory):
                sub_directory = os.path.join(directory, sub_directory)
                if os.path.isdir(sub_directory):
                    for file in os.listdir(sub_directory):
                        _, file_extension = os.path.splitext(file)
                        species_count[specie_name] += 1
                        curr_file_abs_path = os.path.join(sub_directory, file)
                        new_file_abs_path = os.path.join(specie_directory_abs_path,
                                                         "{}{}".format(species_count[specie_name], file_extension))

                        # print(curr_file_abs_path, new_file_abs_path)
                        # shutil.move(curr_file_abs_path, new_file_abs_path)
                        progress_bar.update(species_count[specie_name])
                        # print(species_count)

        print("Moved {} images for \"{}\"".format(species_count[specie_name], specie_name))


def main():
    check_dirs()
    # move_images()
    # clear_images()
    crop_roi_images()


if __name__ == '__main__':
    main()
