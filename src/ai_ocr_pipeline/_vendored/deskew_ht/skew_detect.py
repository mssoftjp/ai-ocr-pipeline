"""Calculates skew angle.

Vendored from https://github.com/ndl-lab/deskew_HT
Original: https://github.com/kakul/Alyn/blob/master/alyn/skew_detect.py
License: MIT (original) + CC BY 4.0 (modifications by NDL Japan)
"""

import numpy as np
import cv2
from skimage.feature import canny
from skimage.transform import hough_line, hough_line_peaks


class SkewDetect:

    piby4 = np.pi / 4

    def __init__(
        self,
        input_file=None,
        output_file=None,
        sigma=0.50,
        display_output=None,
        num_peaks=20,
        skew_max=4.0,
        acc_deg=0.5,
        roi_w=1.0,
        roi_h=1.0,
    ):
        self.sigma = sigma
        self.input_file = input_file
        self.output_file = output_file
        self.display_output = display_output
        self.num_peaks = num_peaks
        self.skew_max = skew_max
        self.acc_deg = acc_deg
        self.roi_w = roi_w
        self.roi_h = roi_h

    def get_max_freq_elem(self, arr):
        max_arr = []
        freqs = {}
        for i in arr:
            if i in freqs:
                freqs[i] += 1
            else:
                freqs[i] = 1

        sorted_keys = sorted(freqs, key=freqs.get, reverse=True)
        max_freq = freqs[sorted_keys[0]]

        for k in sorted_keys:
            if freqs[k] == max_freq:
                max_arr.append(k)

        return max_arr

    def compare_sum(self, value):
        return 44 <= value <= 46

    def calculate_deviation(self, angle):
        angle_in_degrees = np.abs(angle)
        deviation = np.abs(SkewDetect.piby4 - angle_in_degrees)
        return deviation

    def _detect_skew_from_gray(self, img_gray):
        """Core skew detection from a grayscale numpy array."""
        height, width = img_gray.shape
        img = img_gray[
            int(height * (0.5 - self.roi_h / 2.0)):int(height * (0.5 + self.roi_h / 2.0)),
            int(width * (0.5 - self.roi_w / 2.0)):int(width * (0.5 + self.roi_w / 2.0)),
        ]

        img = cv2.resize(img, (img.shape[1] // 4, img.shape[0] // 4))

        edges = canny(img, sigma=self.sigma)
        range_rad = np.arange(
            -np.pi / 2, -np.pi / 2 + np.deg2rad(self.skew_max),
            step=np.deg2rad(self.acc_deg),
        )
        range_rad = np.concatenate([
            range_rad,
            np.arange(-np.deg2rad(self.skew_max), np.deg2rad(self.skew_max),
                       step=np.deg2rad(self.acc_deg)),
        ], axis=0)
        range_rad = np.concatenate([
            range_rad,
            np.arange(np.pi / 2 - np.deg2rad(self.skew_max), np.pi / 2,
                       step=np.deg2rad(self.acc_deg)),
        ], axis=0)

        h, a, d = hough_line(edges, theta=range_rad)

        th = 0.2 * h.max()
        _, ap, _ = hough_line_peaks(h, a, d, threshold=th, num_peaks=self.num_peaks)

        if len(ap) == 0:
            return {
                "Average Deviation from pi/4": 0.0,
                "Estimated Angle": 0.0,
                "Angle bins": [[], [], [], []],
                "Message": "Bad Quality",
            }

        absolute_deviations = [self.calculate_deviation(k) for k in ap]
        average_deviation = np.mean(np.rad2deg(absolute_deviations))
        ap_deg = [np.rad2deg(x) for x in ap]

        for i in range(len(ap_deg)):
            if ap_deg[i] >= 45.0:
                ap_deg[i] -= 90.0
            elif ap_deg[i] <= -45.0:
                ap_deg[i] += 90.0

        bin_0_45 = []
        bin_45_90 = []
        bin_0_45n = []
        bin_45_90n = []

        for ang in ap_deg:
            deviation_sum = 90 - ang + average_deviation
            if self.compare_sum(deviation_sum):
                bin_45_90.append(ang)
                continue
            deviation_sum = ang + average_deviation
            if self.compare_sum(deviation_sum):
                bin_0_45.append(ang)
                continue
            deviation_sum = -ang + average_deviation
            if self.compare_sum(deviation_sum):
                bin_0_45n.append(ang)
                continue
            deviation_sum = 90 + ang + average_deviation
            if self.compare_sum(deviation_sum):
                bin_45_90n.append(ang)

        angles = [bin_0_45, bin_45_90, bin_0_45n, bin_45_90n]
        lmax = 0
        maxi = 0

        for j in range(len(angles)):
            tmp_l = len(angles[j])
            if tmp_l > lmax:
                lmax = tmp_l
                maxi = j

        if lmax:
            ans_arr = self.get_max_freq_elem(angles[maxi])
            ans_res = np.mean(ans_arr)
        else:
            ans_arr = self.get_max_freq_elem(ap_deg)
            ans_res = np.mean(ans_arr)

        return {
            "Average Deviation from pi/4": average_deviation,
            "Estimated Angle": ans_res,
            "Angle bins": angles,
            "Message": "Successfully detected lines",
        }

    def determine_skew_on_memory(self, img_data):
        """Detect skew from a BGR numpy array."""
        img_gray = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)
        return self._detect_skew_from_gray(img_gray)
