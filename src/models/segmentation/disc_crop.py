import operator
from pathlib import Path
import numpy as np
from skimage.measure import label, regionprops
from skimage.transform import resize
from scipy.ndimage import binary_fill_holes
import onnxruntime as ort

MODEL_FILES_DIR = Path(__file__).absolute().parent / 'model_files'
MODEL_PATH = MODEL_FILES_DIR / 'Model_DiscSeg_ORIGA_640.onnx'
SEED = 42

class DiscCrop(object):
    def __init__(self, disc_segmentation_size = 640,
                 margin = 0.25, square_margin = 400) -> None:
        
        self.square_margin = square_margin
        self.margin = margin
        self.disc_segmentation_size = disc_segmentation_size
        
        ort.set_default_logger_severity(4)

        ort.set_seed(SEED)
        self.session = ort.InferenceSession(MODEL_PATH, 
                                            providers=["CUDAExecutionProvider"])
        
    def crop(self, image: np.ndarray):
        if image.dtype == np.uint8:
            image = image.astype(np.float32)
        
        temp_img = resize(image, (self.disc_segmentation_size, self.disc_segmentation_size))
        temp_img = np.expand_dims(temp_img, axis=0)

        disc_map = self.session.run(None, {'input_1:0': temp_img})[0]
        
        disc_map = np.reshape(disc_map, (self.disc_segmentation_size, self.disc_segmentation_size))
        disc_map = BW_img(disc_map)

        try:
            regions = regionprops(label(disc_map))
            minr, minc, maxr, maxc = regions[0].bbox
        except IndexError:
            return square(image, self.square_margin)
        
        minr, minc, maxr, maxc = square_bbox(minr, minc, maxr, maxc, image.shape)
        
        # Calculate scaling factors
        scale_x = image.shape[0] / self.disc_segmentation_size
        scale_y = image.shape[1] / self.disc_segmentation_size
        
        # Resize bounding box coordinates to original image,
        # add margin and clamp
        minr = max(0, int(minr * scale_x * (1 - self.margin)))
        minc = max(0, int(minc * scale_y * (1 - self.margin)))
        maxr = min(image.shape[0], int(maxr * scale_x * (1 + self.margin)))
        maxc = min(image.shape[1], int(maxc * scale_y * (1 + self.margin)))
        
        cropped_image = image[minr:maxr, minc:maxc]
    
        # Crop the bounding box to be square
        squared_image = square(cropped_image)
    
        return squared_image

def square(image, margin = 0):
    assert margin >= 0
    
    height, width, _ = image.shape

    difference = abs(width - height)
    if width > height:
        shape = (max(1, height - margin), max(1, width - difference - margin))
        squared_image = center_crop(image, shape)
    else:
        shape = (max(1, height - difference - margin), max(1, width - margin))
        squared_image = center_crop(image, shape)
    
    return squared_image

# Thresholds
def BW_img(input, thresholding=0.5):
    if input.max() > thresholding:
        binary = input > thresholding
    else:
        binary = input > input.max() / 2.0

    label_image = label(binary)
    regions = regionprops(label_image)
    area_list = [region.area for region in regions]
    if area_list:
        idx_max = np.argmax(area_list)
        binary[label_image != idx_max + 1] = 0
    return binary_fill_holes(np.asarray(binary).astype(int))

def square_bbox(minr, minc, maxr, maxc, shape):
  # Calculate the height and width of the bounding box
  height = maxr - minr
  width = maxc - minc

  # Determine the new size based on the smallest side
  new_size = min(height, width)

  # Calculate the center coordinates of the original bounding box
  center_r = (minr + maxr) // 2
  center_c = (minc + maxc) // 2

  # Update bounding box coordinates to create a square centered at original center
  new_minr = max(0, center_r - new_size // 2)
  new_maxr = min(shape[0], new_minr + new_size)
  new_minc = max(0, center_c - new_size // 2)
  new_maxc = min(shape[1], new_minc + new_size)

  return new_minr, new_minc, new_maxr, new_maxc


def center_crop(img, bounding):
    start = tuple(map(lambda a, da: a//2-da//2, img.shape, bounding))
    end = tuple(map(operator.add, start, bounding))
    slices = tuple(map(slice, start, end))
    return img[slices]

if __name__ == "__main__":
    from PIL import Image
    import matplotlib.pyplot as plt
    import timeit
    import numpy as np
    
    image_path = Path(__file__).absolute().parent.parent.parent.parent / 'data' / 'JustRAIGS' / 'uncropped' / 'TRAIN000000.JPG'
    image = Image.open(image_path)
    image = np.array(image, dtype=np.float32)
    
    disc_crop = DiscCrop(margin=0.1)
    
    cropped_image = disc_crop.crop(image)
    cropped_image = Image.fromarray(cropped_image.astype(np.uint8))
    
    plt.imshow(cropped_image)
    plt.show()
