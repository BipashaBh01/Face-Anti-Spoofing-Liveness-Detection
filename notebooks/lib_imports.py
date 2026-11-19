import os, cv2, json, math, random, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast

import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from facenet_pytorch import MTCNN
from einops import rearrange
from PIL import Image

import shutil

import onnxruntime as ort
from insightface.app import FaceAnalysis


from decord import VideoReader, cpu, gpu

import tempfile
from unittest.mock import patch, MagicMock
import torchvision.transforms.functional as TF

import sys
import logging

from insightface.utils import face_align   # official ArcFace alignment (norm_crop)

import torch
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

import time

from sklearn.model_selection import GroupShuffleSplit

from collections import Counter

import re

from torch.utils.data import WeightedRandomSampler


from torch.utils.data import ConcatDataset

import matplotlib.patches as mpatches

import mediapipe as mp



# ------------------------------------------------------------
# Python standard-library imports
# ------------------------------------------------------------

import hashlib          # Generate hashes for checkpoint/session integrity
import hmac             # Sign and verify challenge-session data
import json             # Save structured session and review information
import logging          # Application and error logging
import secrets          # Cryptographically secure challenge randomisation
import sys              # Python environment information
import time             # Frame timestamps, action timing and timeouts
import uuid             # Unique session identifiers

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Sequence


# ------------------------------------------------------------
# Numerical and computer-vision imports
# ------------------------------------------------------------

import cv2              # Camera capture and image processing
import mediapipe as mp  # Face landmarks, blendshapes and tracking
import numpy as np      # Numerical calculations and landmark geometry


# ------------------------------------------------------------
# Deep-learning imports
# ------------------------------------------------------------

import torch            # Load and run the existing FAS model


# ------------------------------------------------------------
# Signal-processing imports
# ------------------------------------------------------------

from scipy.signal import find_peaks, savgol_filter
from scipy.spatial.transform import Rotation


# ------------------------------------------------------------
# Data-validation imports
# ------------------------------------------------------------

from pydantic import BaseModel, Field


# ------------------------------------------------------------
# Application-interface import
# ------------------------------------------------------------

import gradio as gr     # Build the notebook-based application interface


import inspect

import albumentations as A
from albumentations.pytorch import ToTensorV2