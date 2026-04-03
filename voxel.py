
import cv2
import numpy as np
import open3d as o3d
from pupil_apriltags import Detector


TAG_FAMILY = "tag36h11"
VOXEL_SIZE = 0.02  # 20 mm

# Choose the 3 arena tag IDs that define your region
REGION_TAG_IDS = [0, 1, 2]

# Height limits in robot frame, relative to the table/workspace
Z_MIN = -0.01
Z_MAX = 0.25


def detect_region_tags(image, camera_intrinsic, tag_ids=REGION_TAG_IDS, tag_size=0.055):
    if image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    detector = Detector(families=TAG_FAMILY)

    fx = camera_intrinsic[0, 0]
    fy = camera_intrinsic[1, 1]
    cx = camera_intrinsic[0, 2]
    cy = camera_intrinsic[1, 2]

    tags = detector.detect(
        gray,
        estimate_tag_pose=True,
        camera_params=(fx, fy, cx, cy),
        tag_size=tag_size
    )

    selected = []
    for t in tags:
        if t.tag_id in tag_ids:
            T_cam_tag = np.eye(4)
            T_cam_tag[:3, :3] = t.pose_R
            T_cam_tag[:3, 3] = t.pose_t.flatten()
            selected.append((t.tag_id, T_cam_tag))

    if len(selected) != 3:
        print(f"Need exactly 3 region tags, found {len(selected)}")
        return None

    selected.sort(key=lambda x: x[0])
    return selected


def point_in_triangle_2d(pt, a, b, c):
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

    d1 = sign(pt, a, b)
    d2 = sign(pt, b, c)
    d3 = sign(pt, c, a)

    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

    return not (has_neg and has_pos)


def voxelize_between_three_tags(image, point_cloud, camera_intrinsic, T_cam_robot,
                                region_tag_ids=REGION_TAG_IDS,
                                tag_size=0.055,
                                voxel_size=VOXEL_SIZE,
                                z_min=Z_MIN, z_max=Z_MAX,
                                show=True):
    detected = detect_region_tags(image, camera_intrinsic, region_tag_ids, tag_size)
    if detected is None:
        return None

    T_robot_cam = np.linalg.inv(T_cam_robot)

    # tag centers in robot frame
    tag_centers_robot = []
    for tag_id, T_cam_tag in detected:
        T_robot_tag = T_robot_cam @ T_cam_tag
        tag_centers_robot.append(T_robot_tag[:3, 3])

    a, b, c = [p[:2] for p in tag_centers_robot]

    pts = point_cloud[:, :, :3].reshape(-1, 3)
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    pts = pts[::4]  # light downsample before transform

    pts_m = pts / 1000.0
    ones = np.ones((len(pts_m), 1))
    pts_robot = (T_robot_cam @ np.hstack([pts_m, ones]).T).T[:, :3]

    inside = []
    for p in pts_robot:
        xy = p[:2]
        if point_in_triangle_2d(xy, a, b, c) and (z_min <= p[2] <= z_max):
            inside.append(p)

    inside = np.asarray(inside)
    print(f"Points inside 3-tag region: {len(inside)}")

    if len(inside) < 50:
        print("Not enough points inside triangular tag region.")
        return None

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(inside)

    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
        pcd,
        voxel_size=voxel_size
    )

    if show:
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

        tag_spheres = []
        for center in tag_centers_robot:
            s = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
            s.translate(center)
            s.paint_uniform_color([1, 0, 0])
            tag_spheres.append(s)

        o3d.visualization.draw_geometries(
            [voxel_grid, axes, *tag_spheres],
            window_name="Voxel Grid Between 3 AprilTags",
            width=1280,
            height=720
        )

    return voxel_grid