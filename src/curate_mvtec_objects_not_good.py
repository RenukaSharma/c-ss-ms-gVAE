import torch 
import torchvision.transforms as transforms
from PIL import Image
import os
from PIL import ImageFile
import numpy as np 

ImageFile.LOAD_TRUNCATED_IMAGES=True

random_crop_32 = transforms.RandomCrop((64,64))
resize_64 = transforms.Resize((64,64))
root_dir = "/home1/renuka/AD/mvtec_anomaly_detection_textureANDobject/objects/zipper/test"
# mask_dir = "/home1/renuka/AD/mvtec_anomaly_detection_textureANDobject/textures/grid/ground_truth"
defect_type="squeezed_teeth" # to change
dest_dir = "/home1/renuka/AD/mvtec_AD_dataset/objects-with-novel-ano-in-test/zipper/train"

for file_name in os.listdir(os.path.join(root_dir,defect_type)):
    # print(file_name)
    file_wo_ext = file_name[:-4]
    print(file_wo_ext)
    img = Image.open(os.path.join(root_dir,defect_type, file_name))


    img_cropped = resize_64(img)
    img_cropped.save(os.path.join(dest_dir,"not_good", str(defect_type)+str(file_wo_ext)+".png" ))
