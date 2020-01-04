import os
import sys
import shutil

import progressbar

DATASET_DIR = "../foraminifera-dataset"
IMG_DIR = "img"
OUT_SEGMENTED_IMAGES_DIR = "img/segmented"
OUT_SPECIES_IMAGES_DIR = "img/species"


def clear_images():
    # TODO: clear images, remove small particles and standardize image size
    pass


def check_dirs():
    if not os.path.isdir(DATASET_DIR):
        sys.stderr.write("\"{}\" DOES NOT EXISTS, exiting...".format(DATASET_DIR))
        sys.exit(1)

    for dir_path in [IMG_DIR, OUT_SEGMENTED_IMAGES_DIR, OUT_SPECIES_IMAGES_DIR]:
        if not os.path.isdir(dir_path):
            print("Directory \"{}\" does not exists, creating directory...".format(dir_path))
            os.mkdir(dir_path)


def move_images():
    # first check all directories are ok
    check_dirs()

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
                        new_file_abs_path = os.path.join(specie_directory_abs_path, "{}{}".format(species_count[specie_name], file_extension))

                        # print(curr_file_abs_path, new_file_abs_path)
                        # shutil.move(curr_file_abs_path, new_file_abs_path)
                        progress_bar.update(species_count[specie_name])
                        # print(species_count)

        print("Moved {} images for \"{}\"".format(species_count[specie_name], specie_name))


def main():
    move_images()


if __name__ == '__main__':
    main()