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

frame = zed.image
h, w = frame.shape[:2]
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('hand_feed.mp4', fourcc, 10, (w, h))

print("Recording... Press Ctrl+C to stop.")

try:
    while True:
        frame = zed.image
        if frame is None:
            continue

        bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR) if frame.shape[2] == 4 else frame
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)

        with torch.autocast('cuda', dtype=torch.bfloat16):
            state = processor.set_image(pil_image)
            processor.reset_all_prompts(state)
            state = processor.set_text_prompt(state=state, prompt='hand')

        masks = state['masks']
        scores = state['scores']

        overlay = bgr.copy()
        count = 0
        for i in range(masks.shape[0]):
            if scores[i] < 0.15:
                continue
            mask = masks[i, 0]
            if mask.shape[0] != h or mask.shape[1] != w:
                mask = torch.nn.functional.interpolate(
                    mask.unsqueeze(0).unsqueeze(0).float(),
                    size=(h, w), mode='nearest'
                )[0, 0]
            mask_np = (mask > 0).cpu().numpy().astype(numpy.uint8)
            overlay[mask_np > 0] = [0, 255, 0]
            count += 1

        bgr = cv2.addWeighted(bgr, 0.6, overlay, 0.4, 0)
        out.write(bgr)
        print(f"Hands detected: {count}", end='\r')

except KeyboardInterrupt:
    print("\nStopping...")

finally:
    out.release()
    zed.close()
    print("Saved hand_feed.mp4")