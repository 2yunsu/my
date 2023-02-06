from PIL import Image
im = Image.open('python.png')
size=(32,32)
im.thumbnail(size)
im.save('python-resize.png')