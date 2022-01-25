import csv
import os
import argparse
import cv2
from FrameConstructorCrazy import FrameConstructorCrazy
from ObjectProducer import ObjectProducer
import random as rand
import numpy as np

DATASET_SIZE = 1000
MAX_AMOUNT_OF_OBJECTS = 8
BACKGROUND_PATH = "background.jpg"
SCALE_INCREASE_RATE = 0.1


def gen_dataset(obj_root, dataset_path, size, max_obj_amount):
    with open(f'{dataset_path}/annot.csv', 'w') as annot_file, \
         open(f'{dataset_path}/annot.tensorflow.csv', 'w') as annot_tens_file:
        size_for_each_amount = int(size // max_obj_amount)
        writer = csv.writer(annot_file)
        writer_tens = csv.writer(annot_tens_file)
        background = cv2.imread(BACKGROUND_PATH)

        for obj_amount in range(1, max_obj_amount+1):
            print(f"Generating {obj_amount}:")
            for i in range(size_for_each_amount):
                print(f"{i}/{size_for_each_amount}", end='\r')
                frame, annot, annot_tens= gen_frame(background, obj_root, obj_amount)
                folder_path = os.path.join(dataset_path, str(obj_amount)).replace("\\", "/")
                frame_path = os.path.join(folder_path, f"{obj_amount}_{i}.png").replace("\\", "/")
                annot = np.hstack([np.full(shape=(obj_amount, 1), fill_value=frame_path), annot])
                annot_tens = np.hstack([np.full(shape=(obj_amount, 1), fill_value=f"{obj_amount}_{i}.png"), annot_tens])

                if not os.path.exists(folder_path):
                    os.mkdir(folder_path)
                cv2.imwrite(frame_path, frame)
                writer.writerows(annot)
                writer_tens.writerows(annot_tens)
            print("DONE!     ")



def gen_frame(background, obj_root, obj_amount):

    frame_generated = False
    scale = rand.random()*0.5 + 1.0
    while not frame_generated:
        try:
            frame_const = FrameConstructorCrazy(background, scale=scale, shake_value=1.5)
            obj_prod = ObjectProducer(obj_root)
            for i in range(obj_amount):
                selected_obj, selected_label = obj_prod.get_random_object_image_duplicates()
                frame_const.place_obj(selected_obj, selected_label)
            frame_generated = True
        except ValueError:
            scale += SCALE_INCREASE_RATE


    frame, annotations, annotations_tens = frame_const.generate_frame()

    return frame, annotations, annotations_tens


if __name__ == "__main__":
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True,
                    help="path to file with cropped objects")
    ap.add_argument("-o", "--output", required=True,
                    help="path to folder with dataset")
    args = vars(ap.parse_args())

    if not os.path.exists(args["output"]):
        os.mkdir(args["output"])

    gen_dataset(args['input'], args["output"], DATASET_SIZE, MAX_AMOUNT_OF_OBJECTS)
