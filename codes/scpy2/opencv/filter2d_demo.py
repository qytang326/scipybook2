# -*- coding: utf-8 -*-
from os import path
import cv2
import numpy as np
from traits.api import Array, Float
from traitsui.api import View, Item, VGroup
from .demobase import ImageProcessDemo


class FilterDemo(ImageProcessDemo):
    TITLE = "OpenCV filter2d Demo"
    SETTINGS = ["kernel", "scale"]
    kernel = Array(shape=(3, 3), dtype=np.float)
    scale = Float


    view = View(
        Item("kernel", label="卷积核"),
        Item("scale", label="乘积因子"),
        title="Filter Demo控制面板"
    )

    def control_panel(self):
        return VGroup(
            Item("kernel", label="卷积核"),
            Item("scale", label="乘积因子"),
        )

    def __init__(self, **kwargs):
        super(FilterDemo, self).__init__(**kwargs)
        self.kernel = np.ones((3, 3))
        self.scale = 1.0 / 9.0
        self.connect_dirty("kernel, scale")

    def draw(self):
        kernel = self.kernel * self.scale
        img2 = cv2.filter2D(self.img, -1, kernel)
        self.draw_image(img2)


if __name__ == '__main__':
    demo = FilterDemo()
    demo.configure_traits()
