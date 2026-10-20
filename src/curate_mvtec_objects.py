import torch 
import torchvision.transforms as transforms
from PIL import Image
import os
from PIL import ImageFile
import numpy as np 

ImageFile.LOAD_TRUNCATED_IMAGES=True

random_crop_64 = transforms.RandomCrop((64,64))
resize_64 = transforms.Resize((64,64))
root_dir = "/home1/renuka/AD/mvtec_anomaly_detection_textureANDobject/objects/zipper/test/good"

# defect_type="thread" # to change
dest_dir = "/home1/renuka/AD/mvtec_AD_dataset/objects-with-novel-ano-in-test/zipper/train/good"
for file_name in os.listdir(root_dir):
    # print(file_name)
    file_wo_ext = file_name[:-4]
    print(file_wo_ext)
    img = Image.open(os.path.join(root_dir, file_name))
    for i in range(1):
	    
	    img_cropped = resize_64(img)
	    img_cropped.save(os.path.join(dest_dir, str(file_wo_ext)+"_"+str(i)+"test.png"))