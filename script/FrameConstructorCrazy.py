import random as rand
from Node import Node
import numpy as np
import cv2


class FrameConstructorCrazy:

    def __init__(self, background, scale=1.0, shake_value=2.0, verbose=False):
        self.VERBOSE = verbose
        self.SCALE = scale
        self.SHAKE_VALUE = shake_value
        self.frame = None
        # add alpha channel if background doesn't have it
        if background.shape[-1] == 3:
            background = cv2.cvtColor(background, cv2.COLOR_RGB2RGBA)
            background[:, :, 3] = 255
        self.background = background

        self.height, self.width, self.dim = background.shape
        self.center = np.asarray((self.width // 2, self.height // 2))

        self.root_node = Node(0, 0, self.width, self.height)

    # run through possible placement variants on a frame
    def place_obj(self, img, label):
        scaled_shape = tuple((np.asarray(img.shape[1::-1])//self.SCALE).astype(int))
        img = cv2.resize(img, scaled_shape)
        leaves = self.root_node.find_empty(list())

        # try to place image until successfully placed or can't place it
        while len(leaves) > 0:
            selected_leave = rand.choice(leaves)
            if selected_leave.place_obj(img, label):
                return
            leaves.remove(selected_leave)

        raise ValueError("Object doesn't fit into frame.")



    def overlay_transparent(self, background, object, x, y):
        h, w, _ = object.shape
        b_h = slice(y, h + y)
        b_w = slice(x, w + x)
        o_h = slice(0, h)
        o_w = slice(0, w)

        if y + h > self.height:
            o_h = slice(0, self.height - y)
        if x + w > self.width:
            o_w = slice(0, self.width - x)

        if y < 0:
            o_h = slice(0, h - y)
            h = h - y
        if x < 0:
            o_w = slice(0, w - x)
            w = w - x

        b_h = slice(y, h + y)
        b_w = slice(x, w + x)

        alpha = object[o_h, o_w, 3] / 255


        for channel in range(3):
            background[b_h, b_w, channel] = (1 - alpha) * background[b_h, b_w, channel] + alpha * \
                                                 object[o_h, o_w, channel]

        if self.VERBOSE:
            background = cv2.rectangle(background, (x, y), (x + w, y + h), (255, 0, 0), 1)

        return background

    def calculate_shaken_coords(self, x, y, w, h):
        x_range = (x+w//2)
        y_range = (y+h//2)

        shake_range = self.center - (x_range, y_range)

        if shake_range[0] > 0:
            x_shaken = x + self.SHAKE_VALUE * rand.randint(0, shake_range[0])
        else:
            x_shaken = x + self.SHAKE_VALUE * rand.randint(shake_range[0], 0)
        x_shaken = round(x_shaken)

        if shake_range[1] > 0:
            y_shaken = y + self.SHAKE_VALUE * rand.randint(0, shake_range[1])
        else:
            y_shaken = y + self.SHAKE_VALUE * rand.randint(shake_range[1], 0)
        y_shaken = round(y_shaken)


        if x_shaken > self.width:
            x_shaken = self.width
        if y_shaken > self.height:
            y_shaken = self.height

        if x_shaken < 0:
            x_shaken = rand.randint(-20, 0)
        if y_shaken < 0:
            y_shaken = rand.randint(-20, 0)

        return x_shaken, y_shaken

    # run through tree and place object into frame
    def generate_frame(self):
        nodes = self.root_node.find_images(list())

        self.frame = np.copy(self.background)
        annotations = list()
        annotations_tens = list()

        if self.VERBOSE:
            self.frame = self.root_node.draw_nodes(self.frame)

        for node in nodes:
            img = node.img
            label = node.label

            h, w, _ = img.shape
            x, y = node.x, node.y

            x_shaken, y_shaken = self.calculate_shaken_coords(x, y, w, h)

            # object frame
            if self.VERBOSE:
                self.frame = cv2.rectangle(self.frame, (x, y), (x + w, y + h), (0, 0, 255, 1), thickness=1)
                self.frame = cv2.line(self.frame, self.center, (x_shaken, y_shaken), (255, 0, 0), thickness=1)

            self.frame = self.overlay_transparent(self.frame, img, x_shaken, y_shaken)

            x0_norm = round(x_shaken/self.width, 4)
            y0_norm = round(y_shaken/self.height, 4)
            x1_norm = round((x_shaken+w)/self.width, 4)
            y1_norm = round((y_shaken+h)/self.height, 4)
            annotations.append((label, x0_norm, y0_norm, "", "", x1_norm, y1_norm, "", ""))
            annotations_tens.append((self.width, self.height, label, x_shaken, y_shaken, x_shaken+w, y_shaken+h))

        return self.frame, annotations, annotations_tens


if __name__ == "__main__":
    background = cv2.imread("Picture 707.jpg")
    frame_constr = FrameConstructorCrazy(background, scale=1.4, shake_value=0.7, verbose=True)

    obj1 = cv2.imread("cropped_objects/lipton/Picture 326.png", cv2.IMREAD_UNCHANGED)
    obj2 = cv2.imread("cropped_objects/actimel/Picture 652.png", cv2.IMREAD_UNCHANGED)
    obj3 = cv2.imread("cropped_objects/actimel/Picture 647.png", cv2.IMREAD_UNCHANGED)
    obj4 = cv2.imread("cropped_objects/beer_glass_bottle/Picture 573.png", cv2.IMREAD_UNCHANGED)
    obj5 = cv2.imread("cropped_objects/beer_glass_bottle/Picture 598.png", cv2.IMREAD_UNCHANGED)
    frame_constr.place_obj(obj1, "lipton")
    frame_constr.place_obj(obj2, "actimel")
    frame_constr.place_obj(obj3, "actimel")
    frame_constr.place_obj(obj4, "actimel")
    frame_constr.place_obj(obj5, "actimel")

    cv2.imshow("Frame", frame_constr.generate_frame()[0])
    cv2.waitKey(0)
