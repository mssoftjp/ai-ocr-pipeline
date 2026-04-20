"""Deskews image after detecting skew angle.

Vendored from https://github.com/ndl-lab/deskew_HT
Original: https://github.com/kakul/Alyn/blob/master/alyn/deskew.py
License: MIT (original) + CC BY 4.0 (modifications by NDL Japan)
"""

import numpy as np
import cv2

from ai_ocr_pipeline._vendored.deskew_ht.skew_detect import SkewDetect


class Deskew:

    def __init__(
        self,
        input_file=None,
        output_file=None,
        r_angle=0,
        skew_max=4.0,
        acc_deg=0.1,
        method=1,
        roi_w=1.0,
        roi_h=1.0,
        gray=1.0,
        quality=100,
        short=None,
    ):
        self.input_file = input_file
        self.output_file = output_file
        self.r_angle = r_angle
        self.method = method
        self.gray = gray
        self.quality = quality
        self.short = short
        self.skew_obj = SkewDetect(
            self.input_file,
            skew_max=skew_max,
            acc_deg=acc_deg,
            roi_w=roi_w,
            roi_h=roi_h,
        )

    def deskew_on_memory(self, input_data):
        """Deskew a BGR numpy array in memory.

        Args:
            input_data: numpy array (BGR format, as from cv2.imread)

        Returns:
            Deskewed numpy array (BGR).
        """
        res = self.skew_obj.determine_skew_on_memory(input_data)
        angle = res["Estimated Angle"]
        rot_angle = angle + self.r_angle

        g = self.gray * 255
        rotated = self.rotate_expand(input_data, rot_angle, g)

        if self.short:
            h = rotated.shape[0]
            w = rotated.shape[1]
            if w < h:
                h = int(h * self.short / w + 0.5)
                w = self.short
            else:
                w = int(w * self.short / h + 0.5)
                h = self.short
            rotated = cv2.resize(rotated, (w, h))

        return rotated

    def rotate_expand(self, img, angle=0, g=255):
        h = img.shape[0]
        w = img.shape[1]
        angle_rad = angle / 180.0 * np.pi
        w_rot = int(np.round(
            h * np.absolute(np.sin(angle_rad)) + w * np.absolute(np.cos(angle_rad))
        ))
        h_rot = int(np.round(
            h * np.absolute(np.cos(angle_rad)) + w * np.absolute(np.sin(angle_rad))
        ))
        size_rot = (w_rot, h_rot)
        mat = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        mat[0][2] = mat[0][2] - w / 2 + w_rot / 2
        mat[1][2] = mat[1][2] - h / 2 + h_rot / 2
        rotated = cv2.warpAffine(img, mat, size_rot, borderValue=(g, g, g))
        return rotated
