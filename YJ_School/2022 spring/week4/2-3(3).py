import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

img=Image.open("python-resize.png")
array=np.array(img)

plt.imshow(array)
plt.show()