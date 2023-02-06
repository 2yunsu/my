from PIL import Image
import numpy as np

img_color = Image.open('C:/Users/User/OneDrive/문서/Pycharm/BI Lab/week2/python.png')

img_color=np.array(img_color)

plt.imshow(img_color)
plt.show()