import cv2, numpy, time, torch
from pupil_apriltags import Detector
from xarm.wrapper import XArmAPI
from PIL import Image

from utils.vis_utils import draw_pose_axes
from utils.zed_camera import ZedCamera
from checkpoint0 import get_transform_camera_robot
from checkpoint1 import grasp_cube, place_cube, GRIPPER_LENGTH, CUBE_TAG_FAMILY, CUBE_TAG_ID, CUBE_TAG_SIZE

from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

cube_prompt = 'red cube'
robot_ip = '192.168.1.183'

SAM3_CHECKPOINT = '/home/rob/sam3/checkpoints/sam3.pt'

# SAM 3 responds better to "{color} object" than "{color} cube"
PROMPT_MAP = {
    'red':   'red object',
    'green': 'green object',
    'blue':  'blue object',
}


class CubePoseDetector:
    def __init__(self, camera_intrinsic):
        self.camera_intrinsic = camera_intrinsic
        self.camera_pose = None
        self.detector = Detector(families=CUBE_TAG_FAMILY)

        print("Loading SAM 3 model...")
        
        # TF32 is the sweet spot between FP32 (slow and accurate) and FP16(fast and less accurate)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        self.model = build_sam3_image_model(checkpoint_path=SAM3_CHECKPOINT)
        self.processor = Sam3Processor(self.model, confidence_threshold=0.1)
        print("SAM 3 model loaded.")

    def get_transforms(self, observation, cube_prompt):
        """
        Use SAM 3 to segment the target cube color, detect all AprilTags,
        then match: find the tag with the best overlap across ALL masks.
        """
        # Parse color from prompt
        color = None
        for c in PROMPT_MAP:
            if c in cube_prompt.lower():
                color = c
                break
        if color is None:
            print(f'Could not parse color from prompt: "{cube_prompt}"')
            return None

        sam3_prompt = PROMPT_MAP[color]

        # Prep images
        if len(observation.shape) == 3 and observation.shape[2] == 4:
            bgr = cv2.cvtColor(observation, cv2.COLOR_BGRA2BGR)
            gray = cv2.cvtColor(observation, cv2.COLOR_BGRA2GRAY)
        else:
            bgr = observation
            gray = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)

        h, w = gray.shape

        # Step 1: Run SAM 3
        image_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        with torch.autocast('cuda', dtype=torch.bfloat16):
            state = self.processor.set_image(pil_image)
            self.processor.reset_all_prompts(state)
            state = self.processor.set_text_prompt(state=state, prompt=sam3_prompt)

        all_masks = state['masks']   # (N, 1, H, W)
        scores = state['scores']     # (N,)
        n_masks = all_masks.shape[0]

        print(f"SAM 3 prompt='{sam3_prompt}': {n_masks} detections")
        print(f"  Scores: {[f'{s:.3f}' for s in scores.tolist()]}")

        if n_masks == 0:
            print(f"SAM 3 found nothing for '{sam3_prompt}'")
            return None

        # Convert all masks to numpy
        mask_arrays = []
        for i in range(n_masks):
            m = all_masks[i, 0]
            m_h, m_w = m.shape
            if m_h != h or m_w != w:
                m = torch.nn.functional.interpolate(
                    m.unsqueeze(0).unsqueeze(0).float(),
                    size=(h, w), mode='nearest',
                )[0, 0]
            mask_arrays.append((m > 0).cpu().numpy().astype(numpy.uint8))

        # Step 2: Detect all AprilTags
        fx = self.camera_intrinsic[0, 0]
        fy = self.camera_intrinsic[1, 1]
        cx = self.camera_intrinsic[0, 2]
        cy = self.camera_intrinsic[1, 2]

        tags = self.detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=(fx, fy, cx, cy),
            tag_size=CUBE_TAG_SIZE
        )
        print(f"Detected {len(tags)} AprilTags")

        if len(tags) == 0:
            return None

        # Step 3: For each tag, check overlap with ALL masks
        # Pick the tag+mask combo with the best overlap
        best_tag = None
        best_overlap = 0
        best_mask_idx = -1

        for tag in tags:
            for mask_idx, mask in enumerate(mask_arrays):
                overlap_count = 0
                total_points = 0

                # Check tag center
                cx_tag = int(tag.center[0])
                cy_tag = int(tag.center[1])
                if 0 <= cx_tag < w and 0 <= cy_tag < h:
                    if mask[cy_tag, cx_tag] > 0:
                        overlap_count += 1
                    total_points += 1

                # Check tag corners
                for corner in tag.corners:
                    px = int(numpy.clip(corner[0], 0, w - 1))
                    py = int(numpy.clip(corner[1], 0, h - 1))
                    if mask[py, px] > 0:
                        overlap_count += 1
                    total_points += 1

                # Check points just outside tag (on cube face)
                for corner in tag.corners:
                    direction = corner - tag.center
                    for scale in [1.3, 1.6]:
                        pt = tag.center + scale * direction
                        px = int(numpy.clip(pt[0], 0, w - 1))
                        py = int(numpy.clip(pt[1], 0, h - 1))
                        if mask[py, px] > 0:
                            overlap_count += 1
                        total_points += 1

                if overlap_count > 0:
                    print(f"  Tag {tag.tag_id} x Mask {mask_idx} (score={scores[mask_idx]:.3f}): overlap {overlap_count}/{total_points}")

                mask_score = scores[mask_idx].item()
                if mask_score < 0.15:  # skip low-confidence masks
                    continue
                combined = overlap_count * mask_score
                if combined > best_overlap:
                    best_overlap = combined
                    best_tag = tag
                    best_mask_idx = mask_idx


        if best_tag is None or best_overlap < 0.3:
            print(f'No AprilTag matched any SAM 3 mask for "{cube_prompt}"')
            return None

        print(f"Selected tag {best_tag.tag_id} matched with mask {best_mask_idx} (score={scores[best_mask_idx]:.3f}), overlap={best_overlap}")

        # Step 4: Build transforms
        t_cam_cube = numpy.eye(4)
        t_cam_cube[:3, :3] = best_tag.pose_R
        t_cam_cube[:3, 3] = best_tag.pose_t.flatten()
        t_robot_cube = numpy.linalg.inv(self.camera_pose) @ t_cam_cube

        return t_robot_cube, t_cam_cube


def main():
    zed = ZedCamera()
    camera_intrinsic = zed.camera_intrinsic

    cube_pose_detector = CubePoseDetector(camera_intrinsic)

    arm = XArmAPI(robot_ip)
    arm.connect()
    arm.motion_enable(enable=True)
    arm.set_tcp_offset([0, 0, GRIPPER_LENGTH, 0, 0, 0])
    arm.set_mode(0)
    arm.set_state(0)
    arm.move_gohome(wait=True)
    time.sleep(0.5)

    try:
        cv_image = zed.image
        t_cam_cube = None

        t_cam_robot = get_transform_camera_robot(cv_image, camera_intrinsic)
        if t_cam_robot is None:
            print('Failed to compute cam robot transform')
            return

        cube_pose_detector.camera_pose = t_cam_robot

        result = cube_pose_detector.get_transforms(cv_image, cube_prompt)
        if result is None:
            print(f'Could not detect "{cube_prompt}"')
            return
        t_robot_cube, t_cam_cube = result

       # Visualization
        draw_pose_axes(cv_image, camera_intrinsic, t_cam_cube)
        cv2.imwrite('verify_pose.jpg', cv_image)
        print("Saved verify_pose.jpg — check it, then type 'k' + Enter to execute:")
        if input().strip() != 'k':
            print("Aborted")
            return
        
        grasp_cube(arm, t_robot_cube)
        place_cube(arm, t_robot_cube)

    finally:
        try:
            arm.move_gohome(wait=True)
            time.sleep(0.5)
            arm.disconnect()
        except:
            pass
        try:
            cv2.destroyAllWindows()
        except:
            pass
        try:
            zed.close()
        except:
            pass


if __name__ == "__main__":
    main()