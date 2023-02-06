import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

path = './magazine.npy'

image_pil = np.load(path, allow_pickle=True)
image = np.array(image_pil)
print(image)
# plt.imshow(image)
# plt.show()