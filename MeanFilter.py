import numpy as np;
from  PIL import Image
from numpy.lib.arraypad import pad; 

# Params
FILTER_SIZE = int(3)
IMAGE_PATH = "D:\workspace\IntelligentVision\Filters_MachineVision\Image"
INPUT_IMAGE = IMAGE_PATH + "\TestImage1.jpg"

input_image = np.array(Image.open(INPUT_IMAGE))

def MeanFilter(image, filter_sz):
    # 1. preprocess  of image
    padding_size = int((filter_sz-1)/2)
    pad_input_image = np.pad(image,padding_size,mode="constant")
    #**kwargs  Default is 0.
    print(pad_input_image.shape)
    print("padding_size = ",padding_size)
    m, n = pad_input_image.shape
    output_image = np.copy(pad_input_image)

    # 2. 空间滤波
    filter = np.ones((filter_sz, filter_sz),dtype=int)
    for i in range(padding_size, m-padding_size):
            for j in range(padding_size, n-padding_size):
                output_image[i, j] = np.sum(filter*(pad_input_image[i-padding_size:i+padding_size+1,j-padding_size:j+padding_size+1]))/FILTER_SIZE**2 
    output_image = output_image[padding_size:m-padding_size,padding_size:n-padding_size] #图片裁剪
    print("------------Image Filted---------")
    return output_image

output_image = MeanFilter(input_image, FILTER_SIZE)
output_image = Image.fromarray(output_image)
output_image.save(IMAGE_PATH + "\MeanFilter_TestImage1.jpeg")
print("------------Image Transformed and Saved---------")