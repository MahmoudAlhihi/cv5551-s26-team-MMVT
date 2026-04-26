import cv2
from utils.zed_camera import ZedCamera

zed = ZedCamera()
try:
    print("Grabbing frame...")
    frame = zed.image
    print(f"Got frame: {frame.shape}")
    bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR) if frame.shape[2] == 4 else frame
    cv2.imwrite("speaker_visible.jpg", bgr)
    print(f"Saved speaker_visible.jpg ({bgr.shape[1]}x{bgr.shape[0]})")
finally:
    zed.close()

import cv2, torch, numpy as np
from PIL import Image
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

torch.backends.cuda.matmul.allow_tf32 = True
model = build_sam3_image_model(checkpoint_path='/home/rob/sam3/checkpoints/sam3.pt')
processor = Sam3Processor(model, confidence_threshold=0.05)  # low threshold to see weak hits

# Load test frame: one with the speaker revealed, ideally one without too
img = cv2.imread('speaker_visible.jpg')
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
pil = Image.fromarray(rgb)

prompts = [
    "speaker grille",
    "perforated metal mesh",
    "small circular speaker",
    "bluetooth speaker",
    "metal disc with holes",
    "speaker",
]

with torch.autocast('cuda', dtype=torch.bfloat16):
    state = processor.set_image(pil)
    for p in prompts:
        processor.reset_all_prompts(state)
        state = processor.set_text_prompt(state=state, prompt=p)
        scores = state['scores']
        masks  = state['masks']
        n = masks.shape[0]
        if n == 0:
            print(f"  {p!r:35s} → no detections")
        else:
            top = scores.max().item()
            area = int((masks[scores.argmax(), 0] > 0).sum().item())
            print(f"  {p!r:35s} → {n} hits, top={top:.3f}, area={area}px")