import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

img=Image.open("python-resize.png")
array=np.array(img)
trans=np.transpose(array,(1,0,2))

plt.imshow(trans)###imshow와 show는 무엇이 다른가?
plt.show()