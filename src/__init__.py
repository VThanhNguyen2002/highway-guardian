#!/usr/bin/env python3
"""
Highway Guardian - Source Package

Hệ thống nhận diện biển báo giao thông và phân loại xe
Sử dụng CNN và YOLOv8 cho detection và classification

Author: VThanhNguyen2002
Date: 2025
"""

__version__ = "1.0.0"
__author__ = "VThanhNguyen2002"
__email__ = "your-email@example.com"
__description__ = "Highway Guardian - Traffic Sign and Vehicle Detection System"

# Import main modules
from . import detection
from . import classification
from . import utils

__all__ = [
    "detection",
    "classification", 
    "utils",
    "__version__",
    "__author__",
    "__email__",
    "__description__"
]