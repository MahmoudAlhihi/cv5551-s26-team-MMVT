import numpy as np
import cv2
from PIL import Image
from lang_sam import LangSAM

img = np.load('/tmp/test_image.npy')
if img.shape[2] == 4:
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
else:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img_small = cv2.resize(img, None, fx=0.5, fy=0.5)
pil_img = Image.fromarray(img_small)

model = LangSAM()
masks, boxes, phrases, logits = model.predict(pil_img, 'blue cube')
print(f"Found {len(masks)} masks")