import random as rand

import cv2


class Node:

    def __init__(self, x, y, width, height, img=None, label=None):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

        self.left_child = None
        self.right_child = None

        self.img = None
        self.label = None
        # try to put image into sub leaves if possible, else it'll
        self.place_obj(img, label)

    def place_obj(self, img, label):
        if img is None:
            return False

        img_height, img_width, img_dim = img.shape

        if not self.is_empty():
            return False

        if not self.img_fits(img):
            return False

        if self.height == img_height and self.width == img_width:
            self.img = img
            self.label = label
            return True

        # if image only fits vertically/horizontally, determine the variant
        # else choose randomly
        split_by_height = rand.choice([True, False])
        if self.width == img_width:
            split_by_height = True
        elif self.height == img_height:
            split_by_height = False

        # randomly choose whether image is going to be on top/bottom if split by height,
        # on left/right if split by width
        top_or_left = rand.choice([True, False])

        if split_by_height:
            # split by height
            if top_or_left:
                # put image on top
                self.left_child = Node(self.x, self.y, self.width, img_height, img, label)
                self.right_child = Node(self.x, self.y + img_height, self.width, self.height - img_height)
            else:
                # put image on the bottom
                self.left_child = Node(self.x, self.y, self.width, self.height - img_height)
                self.right_child = Node(self.x, self.y + self.height - img_height, self.width, img_height, img, label)
        else:
            # split by width
            if top_or_left:
                # put image on left
                self.left_child = Node(self.x, self.y, img_width, self.height, img, label)
                self.right_child = Node(self.x + img_width, self.y, self.width - img_width, self.height)
            else:
                # put image on right
                self.left_child = Node(self.x, self.y, self.width - img_width, self.height)
                self.right_child = Node(self.x + self.width - img_width, self.y, img_width, self.height, img, label)

        return True

    def has_children(self):
        return (self.left_child is not None) or (self.right_child is not None)

    def is_empty(self):
        # check if leaf has an image and thus can't be divided
        if self.img is not None:
            return False

        # check if leaf has already been split
        if self.has_children():
            return False

        return True

    # check if image fits into a node, but node is bigger
    def img_fits(self, img):
        img_height, img_width, _ = img.shape

        if (self.width >= img_width) and (self.height >= img_height):
            return True

        return False

    # find leaves
    def find_empty(self, leaves):
        if self.is_empty():
            leaves.append(self)
            return leaves
        elif self.has_children():
            self.left_child.find_empty(leaves)
            self.right_child.find_empty(leaves)

        return leaves

    # find nodes with images
    def find_images(self, nodes):
        if self.img is not None:
            nodes.append(self)
            return nodes
        if self.has_children():
            self.left_child.find_images(nodes)
            self.right_child.find_images(nodes)

        return nodes

    def draw_nodes(self, background):
        background = cv2.rectangle(background, (self.x, self.y), (self.x + self.width, self.y + self.height),
                                   (255, 0, 255, 0.2), thickness=1)
        if self.has_children():
            self.left_child.draw_nodes(background)
            self.right_child.draw_nodes(background)

        return background
