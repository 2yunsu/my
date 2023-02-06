import numpy as np
from PIL import Image
img=Image.open("python.png")
array=np.array(img)
print(array)