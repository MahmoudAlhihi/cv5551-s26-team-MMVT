import numpy as np
from utils.zed_camera import ZedCamera

zed = ZedCamera()
img = zed.image
np.save('/tmp/test_image.npy', img)
print(f"Saved image shape: {img.shape}")
zed.close()
