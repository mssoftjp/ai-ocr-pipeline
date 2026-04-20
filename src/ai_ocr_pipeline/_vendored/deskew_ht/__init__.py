"""Vendored deskew_HT from https://github.com/ndl-lab/deskew_HT

Original code (alyn3) is based on https://github.com/kakul/Alyn (MIT License).
Modifications by National Diet Library, Japan (CC BY 4.0).
See LICENSE_deskew_ht in this directory for full license text.
"""

from ai_ocr_pipeline._vendored.deskew_ht.deskew import Deskew
from ai_ocr_pipeline._vendored.deskew_ht.skew_detect import SkewDetect

__all__ = ["Deskew", "SkewDetect"]
