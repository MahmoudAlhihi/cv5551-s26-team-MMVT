import cv2, torch, numpy
from PIL import Image
from utils.zed_camera import ZedCamera
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

SAM3_CHECKPOINT = '/home/rob/sam3/checkpoints/sam3.pt'

zed = ZedCamera()

print("Loading SAM 3...")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
model = build_sam3_image_model(checkpoint_path=SAM3_CHECKPOINT)
processor = Sam3Processor(model, confidence_threshold=0.1)
print("SAM 3 loaded.")

print("Grabbing frame...")
frame = zed.image
print(f"Got frame: {frame.shape}")

bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR) if frame.shape[2] == 4 else frame
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
pil_image = Image.fromarray(rgb)

print("Running SAM3 inference...")
with torch.autocast('cuda', dtype=torch.bfloat16):
    state = processor.set_image(pil_image)
    processor.reset_all_prompts(state)
    state = processor.set_text_prompt(state=state, prompt='hand')

masks = state['masks']
scores = state['scores']
print(f"Masks: {masks.shape}, Scores: {scores}")

if masks.shape[0] > 0:
    best = scores.argmax().item()
    mask = masks[best, 0]
    h, w = bgr.shape[:2]
    if mask.shape[0] != h or mask.shape[1] != w:
        mask = torch.nn.functional.interpolate(
            mask.unsqueeze(0).unsqueeze(0).float(),
            size=(h, w), mode='nearest'
        )[0, 0]
    mask_np = (mask > 0).cpu().numpy().astype(numpy.uint8)

    overlay = bgr.copy()
    overlay[mask_np > 0] = [0, 255, 0]
    bgr = cv2.addWeighted(bgr, 0.6, overlay, 0.4, 0)
    print(f"Hand detected! Score: {scores[best]:.3f}")
else:
    print("No hand detected — try a different prompt")

cv2.imwrite('hand_test.jpg', bgr)
print("Saved hand_test.jpg")

zed.close()