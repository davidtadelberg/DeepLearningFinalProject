from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt

def resizeImage(image_path):
    desired_size = 512
    im = Image.open(image_path)
    old_size = im.size  # old_size[0] is in (width, height) format

    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    im = im.resize(new_size, Image.ANTIALIAS)
    # create a new image and paste the resized on it

    new_im = Image.new("RGB", (desired_size, desired_size))
    new_im.paste(im, ((desired_size-new_size[0])//2,
                        (desired_size-new_size[1])//2))

    return np.asarray(new_im)


# Test the script.
newImage = resizeImage("test.jpg")
plt.imshow(newImage)
plt.show()