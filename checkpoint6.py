
import cv2, numpy, time, torch
import open3d as o3d
from scipy.spatial.transform import Rotation
from xarm.wrapper import XArmAPI
from PIL import Image

from utils.vis_utils import draw_pose_axes
from utils.zed_camera import ZedCamera
from checkpoint0 import get_transform_camera_robot
from checkpoint1 import grasp_cube, place_cube, GRIPPER_LENGTH
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

CUBE_SIZE = 0.025

robot_ip = '192.168.1.183'
SAM3_CHECKPOINT = '/home/rob/sam3/checkpoints/sam3.pt'

PROMPT_MAP = {
    'red':   'red object',
    'green': 'green object',
    'blue':  'blue object',
}
print("Loading SAM 3 model...")
        
        # TF32 is the sweet spot between FP32 (slow and accurate) and FP16(fast and less accurate)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
_sam3_model = build_sam3_image_model(checkpoint_path=SAM3_CHECKPOINT)
_sam3_processor = Sam3Processor(_sam3_model, confidence_threshold=0.1)
print("SAM 3 model loaded.")



def get_sam3_mask(image_bgr, color='red'):
    
    sam3_prompt = PROMPT_MAP.get(color, 'object')
    h, w = image_bgr.shape[:2]

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)

    with torch.autocast('cuda' , dtype=torch.bfloat16):
        state = _sam3_processor.set_image(pil_image)
        _sam3_processor.reset_all_prompts(state)
        state = _sam3_processor.set_text_prompt(state = state, prompt = sam3_prompt)

    all_masks = state['masks']
    scores = state['scores']
    n_masks = all_masks.shape[0]

    print(f"SAM 3 prompt='{sam3_prompt}': {n_masks} detections")
    print(f"  Scores: {[f'{s:.3f}' for s in scores.tolist()]}")

    if n_masks == 0:
        print(f"SAM 3 found nothing for '{sam3_prompt}'")
        return None
    
    valid = scores > 0.15
    if not valid.any():
        print("No masks above confidence threshold")
        return None
    
    best_idx = scores[valid].argmax()
    valid_indices = torch.where(valid)[0]
    best_original_idx = valid_indices[best_idx].item()
    print(f"Best mask: idx={best_original_idx}, score={scores[best_original_idx]:.3f}")
    
    m = all_masks[best_original_idx, 0]
    m_h, m_w = m.shape
    if m_h != h or m_w != w:
        m = torch.nn.functional.interpolate(
            m.unsqueeze(0).unsqueeze(0).float(),
            size=(h, w), mode='nearest',
        )[0, 0]
    return (m > 0).cpu().numpy().astype(numpy.uint8)







def get_transform_cube(observation, camera_intrinsic, camera_pose, color = 'red'):
    """
    Calculate the transformation matrix for the cube relative to the robot base frame, 
    as well as relative to the camera frame.

    This function leverages text prompts to semantically segment a specific 
    cube (e.g., 'red cube') and determines the cube's pose using its 3D point cloud.

    Parameters
    ----------
    observation : list or tuple
        A collection containing [image, point_cloud], where image is the 
        RGB/BGRA array and point_cloud is the registered 3D point cloud.
    camera_intrinsic : numpy.ndarray
        The 3x3 intrinsic camera matrix.
    camera_pose : numpy.ndarray
        A 4x4 transformation matrix representing the camera's pose in the robot base frame (t_cam_robot).
        All translations are in meters.

    Returns
    -------
    tuple or None
        If successful, returns a tuple (t_robot_cube, t_cam_cube) where both 
        are 4x4 transformation matrices with translations in meters. 
        If no matching object is segmented, returns None.
    """
    image, point_cloud = observation

    # TODO
    # semantic segmentation (red Mask)
    # red spans the 0 and 180 boundaries in hsv
    if image.shape[2] == 4:
        bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    else:
        bgr = image

    mask = get_sam3_mask(bgr, color=color)
    if mask is None:
        print(f"SAM 3 failed to segment '{color}' cube")
        return None
    mask_area = mask.sum()
    print(f"SAM 3 mask area: {mask_area} pixels")
    if mask_area < 100:
        print("Mask too small")
        return None
    

    kernel= numpy.ones((5,5), numpy.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # Shrink the mask slightly to avoid edge noise
    kernel = numpy.ones((3,3), numpy.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    

    # filtering NaNs and extracting points
    # point_cloud is (H, W, 4) -> [x, y, z, color]
    raw_points = point_cloud[mask > 0][:, :3]
    
    # removing invalid depth readings (NaN or Inf)
    cpoints = raw_points[numpy.all(numpy.isfinite(raw_points), axis=1)]

    if len(cpoints) < 50:
        print("Not enough valid points detected")
        return None
    print(f"Point cloud sample: {point_cloud[600, 1100, :3]}")

    # using Open3D for Oriented Bounding Box
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cpoints)

    # interpret the camera pose
    T_robot_cam = camera_pose  
    T_cam_robot = numpy.linalg.inv(T_robot_cam) # Use this to move points to Robot
    
    # get center and rotation from the geometry
    cpoints_m = cpoints / 1000.0
    ones = numpy.ones((len(cpoints_m), 1))
    cpoints_robot = (T_cam_robot @ numpy.hstack([cpoints_m, ones]).T).T[:, :3]

    # find the top Surface
    z_max = numpy.max(cpoints_robot[:, 2])
    top_mask = cpoints_robot[:, 2] > (z_max - 0.005) 
    top_points = cpoints_robot[top_mask]

    if len(top_points) < 10:
        return None

    # calc XY Center and yaw
    pts_2d_top = top_points[:, :2].astype(numpy.float32)
    rect = cv2.minAreaRect(pts_2d_top)
    (center_xy, (w, h), angle_deg) = rect

    # handle orientation (stand the long/short edge)
    if w < h:
        angle_deg += 90.0
    yaw_robot = numpy.deg2rad(angle_deg)

    # 4calc True Geometric Center
    # the center Z is the top surface Z minus half the cube height
    center_z = z_max - (CUBE_SIZE / 2.0)

    # build transform matrix
    Rz_robot = numpy.array([
        [numpy.cos(yaw_robot), -numpy.sin(yaw_robot), 0],
        [numpy.sin(yaw_robot),  numpy.cos(yaw_robot), 0],
        [0,                     0,                    1]
    ]) @ numpy.diag([1, -1, -1]) # maintain gripper down orientation

    t_robot_cube = numpy.eye(4)
    t_robot_cube[:3, :3] = Rz_robot
    t_robot_cube[:3, 3] = [center_xy[0], center_xy[1], center_z]

    t_cam_cube = T_robot_cam @ t_robot_cube

    return t_robot_cube, t_cam_cube

def main():

    # Initialize ZED Camera
    zed = ZedCamera()
    camera_intrinsic = zed.camera_intrinsic

    # Initialize Lite6 Robot
    arm = XArmAPI(robot_ip)
    arm.connect()
    arm.motion_enable(enable=True)
    arm.set_tcp_offset([0, 0, GRIPPER_LENGTH, 0, 0, 0])
    arm.set_mode(0)
    arm.set_state(0)
    arm.move_gohome(wait=True)
    time.sleep(0.5)

    try:
        # Get Observation
        cv_image = zed.image
        point_cloud = zed.point_cloud

        t_cam_cube = None
        # TODO
        t_cam_robot = get_transform_camera_robot(cv_image, camera_intrinsic)
        if t_cam_robot is None:
            print("Failed to detect registration tags")
            return

        # detecting cube using pure vis
        result = get_transform_cube([cv_image, point_cloud], camera_intrinsic, t_cam_robot)
        
        if result is None:
            print("Cube detection failed")
            return
            
        t_robot_cube, t_cam_cube = result
        
        # Visualization
        draw_pose_axes(cv_image, camera_intrinsic, t_cam_cube)
        cv2.imwrite('verify_pose_cp6.jpg', cv_image)
        print("Saved verify_pose_cp6.jpg — check it, then type 'k' + Enter to execute:")
        if input().strip() != 'k':
            print("Aborted")
            return


            # TODO
            # grasp
        grasp_cube(arm, t_robot_cube)
        time.sleep(0.5)
            # place
        place_cube(arm, t_robot_cube)
            # return home
        arm.move_gohome(wait=True)
    
    finally:
        # Close Lite6 Robot
        arm.move_gohome(wait=True)
        time.sleep(0.5)
        arm.disconnect()

        # Close ZED Camera
        zed.close()

if __name__ == "__main__":
    main()