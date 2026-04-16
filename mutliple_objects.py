import cv2, torch, numpy, time
from PIL import Image
from utils.zed_camera import ZedCamera
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

SAM3_CHECKPOINT = '/home/rob/sam3/checkpoints/sam3.pt'

CLASSES = {
    'red cup':  [0, 0, 255],    # red
    'blue cube': [255, 0, 0],    # blue
    'smartphone':   [0, 255, 0],    # red
    'large tablet' : [128,0, 128],
}

zed = ZedCamera()

print("Loading SAM 3...")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
model = build_sam3_image_model(checkpoint_path=SAM3_CHECKPOINT)
processor = Sam3Processor(model, confidence_threshold=0.1)
print("SAM 3 loaded.")

frame = zed.image
h, w = frame.shape[:2]

print(f"Recording with classes: {list(CLASSES.keys())}")
print("Press Ctrl+C to stop.")

frames = []
start = time.time()

try:
    while True:
        frame = zed.image
        if frame is None:
            continue

        bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR) if frame.shape[2] == 4 else frame
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        overlay = bgr.copy()

        for prompt, color in CLASSES.items():
            with torch.autocast('cuda', dtype=torch.bfloat16):
                state = processor.set_image(pil_image)
                processor.reset_all_prompts(state)
                state = processor.set_text_prompt(state=state, prompt=prompt)

            masks = state['masks']
            scores = state['scores']

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
                overlay[mask_np > 0] = color

        bgr = cv2.addWeighted(bgr, 0.6, overlay, 0.4, 0)
        frames.append(bgr.copy())
        print(f"Frames captured: {len(frames)}", end='\r')

except KeyboardInterrupt:
    print("\nStopping...")

finally:
    elapsed = time.time() - start
    actual_fps = max(1, len(frames) / elapsed) if elapsed > 0 else 1
    print(f"Captured {len(frames)} frames in {elapsed:.1f}s ({actual_fps:.2f} FPS)")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('multi_detect.mp4', fourcc, actual_fps, (w, h))
    for f in frames:
        out.write(f)
    out.release()
    zed.close()
    print("Saved multi_detect.mp4")