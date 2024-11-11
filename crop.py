import os

import glob
from pathlib import Path

import numpy as np
from tqdm import trange
from src.models.segmentation import DiscCrop
from PIL import Image

DATA_DIR = Path(__file__).absolute().parent / 'data' / 'JustRAIGS'

if __name__ == '__main__':
    
    cropped_dir = DATA_DIR / 'cropped'
    uncropped_dir = DATA_DIR / 'uncropped'

    # Get file names without extension from cropped images
    cropped_file_names = {Path(path).stem.split('_')[0] for path in glob.glob(str(cropped_dir / '*.JPG'))}
    
    # Get file names without extension from uncropped images
    uncropped_file_names = {Path(path).stem for path in glob.glob(str(uncropped_dir / '*.JPG'))}
    
    # Remove cropped file names from uncropped file names
    uncropped_file_names -= cropped_file_names
    
    # Convert remaining file names back to paths if necessary
    uncropped_paths = [uncropped_dir / (file_name + '.JPG') for file_name in uncropped_file_names]
    already_cropped = len(cropped_file_names)
    
    disc_crop = DiscCrop()
    
    with trange(len(uncropped_paths)) as progress_bar:
        for index, path in zip(progress_bar, uncropped_paths):
            image_id = os.path.basename(path).split('.')[0]
            
            image = Image.open(path)
            image = np.array(image, dtype=np.float32)
            
            cropped_image = disc_crop.crop(image)
            
            cropped_image = Image.fromarray(cropped_image.astype(np.uint8))
            cropped_image.save(cropped_dir / f"{image_id}_cropped.JPG")