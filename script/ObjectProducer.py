import os
import random as rand
import cv2

class ObjectProducer:

    def __init__(self, obj_root):
        self.ROOT = obj_root
        self.obj_names = os.listdir(obj_root)
        self.obj_variants = {obj_name: os.listdir(os.path.join(obj_root, obj_name)) for obj_name in  self.obj_names}

    def get_random_object_image(self):
        if len(self.obj_names) <= 0:
            raise RuntimeError("Unable to retrieve new objects")
        obj_name = rand.choice(self.obj_names)
        obj_variant = rand.choice(self.obj_variants[obj_name])

        obj_path = os.path.join(self.ROOT, obj_name, obj_variant)

        self.obj_names.remove(obj_name)

        return cv2.imread(obj_path, cv2.IMREAD_UNCHANGED), obj_name

    def get_random_object_image_duplicates(self):
        if len(self.obj_names) <= 0:
            raise RuntimeError("Unable to retrieve new objects")
        obj_name = rand.choice(self.obj_names)
        obj_variant = rand.choice(self.obj_variants[obj_name])

        obj_path = os.path.join(self.ROOT, obj_name, obj_variant)

        return cv2.imread(obj_path, cv2.IMREAD_UNCHANGED), obj_name

