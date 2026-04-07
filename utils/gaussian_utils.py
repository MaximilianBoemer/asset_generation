import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation as R
import os
import json

def transform_gaussians(xyz, scales, rots, transform_mat, t_xyz, scaling_factor):
    """
    Transforms gaussian parameters using a 3x3 matrix.
    - transform_mat is 3x3 (rotation or reflection).
    - t_xyz is [x, y, z] to translate.
    - scaling_factor: amount to scale the whole scene.
    """
    # det of transformation
    det = np.linalg.det(transform_mat)
    
    # Transform positions
    xyz_new = (transform_mat @ (xyz * scaling_factor).T).T + t_xyz
    
    # Transform scales
    scales_new = scales + np.log(scaling_factor)
    
    # Transform quaternions
    # rots in ply are w,x,y,z -> scipy xyzw
    r_current = R.from_quat(np.column_stack([rots[:, 1], rots[:, 2], rots[:, 3], rots[:, 0]]))
    r_current_mats = r_current.as_matrix() # (N, 3, 3)
    
    # new_mats = transform_mat @ r_current_mats
    new_mats = np.matmul(transform_mat, r_current_mats)
    
    # If the transformation is a reflection (det < 0), we fix the handedness
    # for the quaternion by flipping one of the Gaussian's axes.
    # Since Gaussians are symmetric, this doesn't change the shape.
    if det < 0:
        new_mats[:, :, 0] *= -1
        
    r_new = R.from_matrix(new_mats)
    r_new_quat = r_new.as_quat() # xyzw
    
    # Back to w, x, y, z
    rots_new = np.column_stack([r_new_quat[:, 3], r_new_quat[:, 0], r_new_quat[:, 1], r_new_quat[:, 2]])
    
    return xyz_new, scales_new, rots_new

def lightweight_load_and_transform(
    ply_path: str,
    transform_mat: np.ndarray,
    t_xyz: np.ndarray,
    scaling_factor: float,
    gaussian_type: str = "2D",
    spotiness_correction_factor: float = 0.2
):
    plydata = PlyData.read(ply_path)
    vertex = plydata['vertex']
    
    prop_names = [p.name for p in vertex.properties]
    
    xyz = np.vstack([vertex['x'], vertex['y'], vertex['z']]).T
    
    scale_names = [p for p in prop_names if p.startswith('scale_')]
    scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
    scales = np.vstack([vertex[k] for k in scale_names]).T
    
    rot_names = [p for p in prop_names if p.startswith('rot_')]
    rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
    rots = np.vstack([vertex[k] for k in rot_names]).T
    
    xyz_new, scales_new, rots_new = transform_gaussians(xyz, scales, rots, transform_mat, t_xyz, scaling_factor)
    
    xyz_new = np.ascontiguousarray(xyz_new)
    scales_new = np.ascontiguousarray(scales_new)
    rots_new = np.ascontiguousarray(rots_new)
    
    new_dtype = []
    
    if gaussian_type == "2D":
        if 'scale_2' in scale_names:
            scale_names.remove('scale_2')
        scales_new = scales_new[:, :2] * spotiness_correction_factor
        opacity = vertex['opacity'] / (spotiness_correction_factor ** 2)
    else:
        opacity = vertex['opacity']
        
    for p in vertex.properties:
        if p.name == 'x': new_dtype.append(('x', 'f4'))
        elif p.name == 'y': new_dtype.append(('y', 'f4'))
        elif p.name == 'z': new_dtype.append(('z', 'f4'))
        elif p.name in scale_names: new_dtype.append((p.name, 'f4'))
        elif p.name in rot_names: new_dtype.append((p.name, 'f4'))
        elif p.name == 'opacity': new_dtype.append(('opacity', 'f4'))
        elif p.name == 'scale_2' and gaussian_type == "2D": continue
        elif p.name.startswith('f_rest_'):
            new_dtype.append((p.name, 'f4'))
        else:
            new_dtype.append((p.name, p.val_dtype))
            
    new_elements = np.empty(len(vertex), dtype=new_dtype)
    
    for i, name in enumerate(['x', 'y', 'z']):
        new_elements[name] = xyz_new[:, i]       
    for i, name in enumerate(scale_names):
        new_elements[name] = scales_new[:, i]        
    for i, name in enumerate(rot_names):
        new_elements[name] = rots_new[:, i]
        
    new_elements['opacity'] = opacity
    
    for p in prop_names:
        if p in ['x', 'y', 'z', 'opacity'] or p in scale_names or p in rot_names or (p == 'scale_2' and gaussian_type == '2D'):
            continue
        if p.startswith('f_rest_'):
            new_elements[p] = np.zeros_like(vertex[p])
        else:
            new_elements[p] = vertex[p]            
    return new_elements, xyz_new

def process_and_save_gaussians(folder_path, transform_mat, t_xyz, scaling_factor, spotiness_correction_factor):
    base_ply = os.path.join(folder_path, "extracted_gaussians.ply")
    
    # 1. Generate 3D Gaussians (with initial t_xyz, usually [0,0,0])
    print(f"Generating 3D Gaussians...")
    elements_3d, xyz_3d = lightweight_load_and_transform(
        base_ply, transform_mat, t_xyz, scaling_factor, "3D", spotiness_correction_factor
    )
    
    # 2. Calculate centering offset in world space
    min_xyz = np.min(xyz_3d, axis=0)
    max_xyz = np.max(xyz_3d, axis=0)
    t_xyz_center = -0.5 * (min_xyz + max_xyz)
    
    # Apply centering to the already-transformed 3D positions
    elements_3d['x'] += t_xyz_center[0]
    elements_3d['y'] += t_xyz_center[1]
    elements_3d['z'] += t_xyz_center[2]
    
    xyz_centered = xyz_3d + t_xyz_center
    min_xyz_centered = np.min(xyz_centered, axis=0)
    max_xyz_centered = np.max(xyz_centered, axis=0)
    dims = max_xyz_centered - min_xyz_centered
    print(f"Centered Bounds - Min: {min_xyz_centered}, Max: {max_xyz_centered}, Dims: {dims}")
    
    with open(os.path.join(folder_path, "dims.json"), "w") as f:
        json.dump({"length": float(dims[0]), "width": float(dims[1]), "height": float(dims[2])}, f)
        
    el_3d = PlyElement.describe(elements_3d, 'vertex')
    PlyData([el_3d], text=False).write(os.path.join(folder_path, "gaussians_3d.ply"))
    print("Saved gaussians_3d.ply and dims.json")
    
    # 3. Generate 2D Gaussians (using the same t_xyz_center logic)
    print(f"Generating 2D Gaussians...")
    elements_2d, _ = lightweight_load_and_transform(
        base_ply, transform_mat, t_xyz, scaling_factor, "2D", spotiness_correction_factor
    )
    
    # Apply the same centering offset
    elements_2d['x'] += t_xyz_center[0]
    elements_2d['y'] += t_xyz_center[1]
    elements_2d['z'] += t_xyz_center[2]
    
    el_2d = PlyElement.describe(elements_2d, 'vertex')
    PlyData([el_2d], text=False).write(os.path.join(folder_path, "gaussians_2d.ply"))
    print("Saved gaussians_2d.ply")
