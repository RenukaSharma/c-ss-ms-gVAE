import os
from tqdm import tqdm
from PIL import Image
import numpy as np
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES=True

img_folder = '/home1/renuka/AD_CorrosionDataset/data_dir_patch-based/test/Images'
for file_name in tqdm(os.listdir(img_folder)):
    img = Image.open(os.path.join(img_folder, file_name))
    img.save('/home1/renuka/AD_CorrosionDataset/data_dir_patch-based/test/png/'+file_name[:-4]+ ".png","PNG")
