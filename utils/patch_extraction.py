# FORCE REFRESH: Updated at 2026-03-21 09:05
import numpy as np
from scipy.spatial.transform import Rotation
from PIL import Image, ImageDraw
from typing import Tuple, List, Optional, Any, Union


def get_corners(
    row: Any, 
    x_col: str = "center_x", 
    y_col: str = "center_y", 
    z_col: str = "center_z", 
    lx_col: str = "size_x", 
    ly_col: str = "size_y", 
    lz_col: str = "size_z",
    qx_col: str = "orientation_x", 
    qy_col: str = "orientation_y", 
    qz_col: str = "orientation_z", 
    qw_col: str = "orientation_w"
) -> np.ndarray:
    """
    Compute 8 corners of a 3D bounding box in the world frame.
    
    Args:
        row: A pandas Series or dict containing bbox parameters.
        x_col, y_col, z_col: Center coordinates.
        lx_col, ly_col, lz_col: Full dimensions (size).
        qx_col, qy_col, qz_col, qw_col: Orientation quaternion (x,y,z,w).
        
    Returns:
        np.ndarray: (8, 3) matrix of coordinate in world frame.
    """
    c = np.array([float(row[x_col]), float(row[y_col]), float(row[z_col])])
    s = np.array([float(row[lx_col]), float(row[ly_col]), float(row[lz_col])]) / 2.0
    
    # Get rotation matrix from quaternion (standard scipy order is [x, y, z, w])
    q = [float(row[qx_col]), float(row[qy_col]), float(row[qz_col]), float(row[qw_col])]
    R = Rotation.from_quat(q).as_matrix()
    
    # 8 corners locally
    local = np.array([
        [ s[0],  s[1],  s[2]], [ s[0],  s[1], -s[2]], [ s[0], -s[1],  s[2]], [ s[0], -s[1], -s[2]],
        [-s[0],  s[1],  s[2]], [-s[0],  s[1], -s[2]], [-s[0], -s[1],  s[2]], [-s[0], -s[1], -s[2]]
    ])
    return (R @ local.T).T + c

def project(
    corners: np.ndarray, 
    corners_ts: int,
    cam_ts: int, 
    cam_id: str, 
    egomotion: Any, 
    extrinsics: Any, 
    intrinsics: Any,
    image_size: Optional[Tuple[int, int]] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Project 3D corners (local to rig at corners_ts) to 2D camera pixels at cam_ts.
    
    Returns:
        uv: (N, 2) pixel coordinates (scaled if image_size is provided)
        mask: (N,) bool mask for corners INSIDE the image and in front
        in_front: (N,) bool mask for corners in front of the camera
    """
    model = intrinsics.camera_models[cam_id]
    p_label = egomotion(corners_ts).pose
    p_img = egomotion(cam_ts).pose
    cam_p = extrinsics.sensor_poses[cam_id]
    
    # Rigid Transformation: Local(ts_label) -> World -> Vehicle(ts_img) -> Camera
    corners_world = p_label.apply(corners)
    corners_cam = (p_img * cam_p).inv().apply(corners_world)
    
    in_front = corners_cam[:, 2] > 0
    uv = model.ray2pixel(corners_cam)
    # mask for patch extraction (must be inside image)
    mask = ~model.is_out_of_bounds(uv) & in_front
    
    if image_size is not None:
        scale_x = image_size[0] / model.width
        scale_y = image_size[1] / model.height
        uv = uv * np.array([scale_x, scale_y])
        
    return uv, mask, in_front

def get_patch_rect(uv: np.ndarray, mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Compute integer bbox for valid projected pixels.
    """
    v = uv[mask]
    if len(v) == 0: return None
    return (int(np.floor(v[:, 0].min())), int(np.floor(v[:, 1].min())), 
            int(np.ceil(v[:, 0].max())), int(np.ceil(v[:, 1].max())))

def draw_bbox_3d(
    image: Image.Image, 
    uv: np.ndarray, 
    in_front: np.ndarray, 
    color: str = "lime", 
    line_width: int = 2
) -> Image.Image:
    """
    Draw 3D bounding box wireframe on a PIL image.
    Edge order: [front/back, top/bottom, left/right]
    """
    draw = ImageDraw.Draw(image)
    edges = [
        (0,1), (0,2), (0,4), (1,3), (1,5), (2,3), (2,6), (3,7), (4,5), (4,6), (5,7), (6,7)
    ]
    for i, j in edges:
        # Drawing if both are in front, letting PIL handle image-edge clipping
        if in_front[i] and in_front[j]:
            draw.line([tuple(uv[i]), tuple(uv[j])], fill=color, width=line_width)
    return image
