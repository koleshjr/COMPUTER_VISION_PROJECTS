# Re-declare/import the AlbumentationsTransform class
from typing import Dict
from io import BytesIO
import numpy as np
from PIL import Image
from fastai.vision.all import *
from contextlib import contextmanager
import pathlib

# Fix the windows os issue
@contextmanager
def set_posix_windows():
    posix_backup = pathlib.PosixPath
    try:
        pathlib.PosixPath = pathlib.WindowsPath
        yield
    finally:
        pathlib.PosixPath = posix_backup

# read the image 
def read_image(file: bytes) -> PILImage:
    return file

 

#predict function
def predict_sign(image) -> Dict:      
    EXPORT_PATH = pathlib.Path("models/ksl_model_improved.pkl")

    with set_posix_windows():
        learn = load_learner(EXPORT_PATH)
        labels = learn.dls.vocab     
        _, _, probs = learn.predict(image)
        max_idx = torch.argmax(probs).item()
        return {
            "prediction": labels[max_idx],
            "probability": probs[max_idx],
        }