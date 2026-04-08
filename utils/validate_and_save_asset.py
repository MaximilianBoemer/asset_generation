import gradio as gr
import trimesh
import os
import sys
import json
import numpy as np
from scipy.spatial.transform import Rotation as R
from plyfile import PlyData

from utils.gaussian_utils import process_and_save_gaussians

def guess_gaussian_transform(xyz, dims_json_path):
    if not os.path.exists(dims_json_path):
        raise FileNotFoundError(f"dims.json not found at {dims_json_path}")
        
    with open(dims_json_path, 'r') as f:
        target_dims = json.load(f)
        
    p_min, p_max = np.min(xyz, axis=0), np.max(xyz, axis=0)
    orig_dims_vec = p_max - p_min
    
    # Match target dimensions to mesh dimensions using ratios
    target_vec = np.array([target_dims['length'], target_dims['width'], target_dims['height']])
    target_ratios = target_vec / np.sum(target_vec)
    
    from itertools import permutations
    best_p = None
    min_err = float('inf')
    for p in permutations([0, 1, 2]):
        perm_dims = orig_dims_vec[list(p)]
        perm_ratios = perm_dims / np.sum(perm_dims)
        err = np.linalg.norm(perm_ratios - target_ratios)
        if err < min_err:
            min_err = err
            best_p = p
            
    mat = np.zeros((3, 3))
    mat[0, best_p[0]] = 1.0 # Target X (Length) -> Raw best_p[0]
    mat[1, best_p[1]] = 1.0 # Target Y (Width) -> Raw best_p[1]
    mat[2, best_p[2]] = 1.0 # Target Z (Height) -> Raw best_p[2]
    
    # Enforce a proper rotation (determinant must be 1.0)
    if np.linalg.det(mat) < 0:
        mat[2, best_p[2]] = -1.0
    
    scaling_factor = target_dims['length'] / orig_dims_vec[best_p[0]]
    print(f"\n--- Diagnostic: Gaussian Initial State ---")
    print(f"PLY raw bounds: {p_min} to {p_max}")
    print(f"PLY raw dims:   {orig_dims_vec}")
    print(f"Target dims:    {target_vec}")
    print(f"Initial Guess Matrix:\n{mat}")
    
    return mat, scaling_factor

class ValidationApp:
    def __init__(self, folder_path, spotiness_correction_factor):
        self.folder_path = folder_path
        self.spotiness_correction_factor = spotiness_correction_factor
        
        self.base_ply_path = os.path.join(folder_path, "extracted_gaussians.ply")
        self.base_glb_path = os.path.join(folder_path, "extracted_mesh.glb")
        self.dims_json_path = os.path.join(os.path.dirname(folder_path), "dims.json")
        
        # UI State
        self.preview_mode = "Gaussians" # Default
        
        print("Loading Gaussian PLY for preview...")
        plydata = PlyData.read(self.base_ply_path)
        vertex = plydata['vertex']
        self.raw_xyz = np.vstack([vertex['x'], vertex['y'], vertex['z']]).T
        
        # Color extraction
        try:
            SH_C0 = 1 / (2 * np.sqrt(np.pi))
            colors = np.stack([vertex['f_dc_0'], vertex['f_dc_1'], vertex['f_dc_2']], axis=-1)
            colors = 0.5 + SH_C0 * colors
            self.raw_colors = np.clip(colors * 255, 0, 255).astype(np.uint8)
        except Exception:
            self.raw_colors = None
            
        # Initial guess
        self.current_matrix, self.scaling_factor = guess_gaussian_transform(self.raw_xyz, self.dims_json_path)
        
        # Downsample for preview
        num_points = len(self.raw_xyz)
        if num_points > 30000:
            indices = np.random.choice(num_points, 30000, replace=False)
            self.preview_xyz = self.raw_xyz[indices]
            self.preview_colors = self.raw_colors[indices] if self.raw_colors is not None else None
        else:
            self.preview_xyz = self.raw_xyz
            self.preview_colors = self.raw_colors

    def _apply_transform_inc(self, inc_matrix):
        """Apply an incremental 3x3 matrix to the current transformation (extrinsic/world-space view)"""
        if inc_matrix is not None:
            self.current_matrix = inc_matrix @ self.current_matrix
        return self._generate_preview_glb()

    def _generate_preview_glb(self):
        """Generates a preview point cloud or mesh GLB"""
        # Transform logic (always synced to Gaussian points as master)
        xyz_transformed_master = (self.current_matrix @ (self.raw_xyz * self.scaling_factor).T).T
        min_xyz = np.min(xyz_transformed_master, axis=0)
        max_xyz = np.max(xyz_transformed_master, axis=0)
        t_xyz_center = -0.5 * (min_xyz + max_xyz)
        
        if self.preview_mode == "Gaussians":
            xyz_centered = (self.current_matrix @ (self.preview_xyz * self.scaling_factor).T).T + t_xyz_center
            pc = trimesh.PointCloud(vertices=xyz_centered, colors=self.preview_colors)
            base_preview = pc
        else:
            # Mesh Mode
            mesh = trimesh.load(self.base_glb_path, process=False)
            transform = np.eye(4)
            transform[:3, :3] = self.current_matrix * self.scaling_factor
            mesh.apply_transform(transform)
            mesh.apply_translation(t_xyz_center)
            base_preview = mesh
            
        # Helpers
        axis_mesh = trimesh.creation.axis(origin_size=0.05, axis_radius=0.02, axis_length=5.0)
        grid_lines = []
        for i in range(-5, 6):
            grid_lines.append([[i, -5, 0], [i, 5, 0]])
            grid_lines.append([[-5, i, 0], [5, i, 0]])
        grid = trimesh.load_path(grid_lines)
        
        scene = trimesh.Scene([base_preview, axis_mesh, grid])
        tmp_path = os.path.join(self.folder_path, f"preview_{self.preview_mode.lower()}.glb")
        scene.export(tmp_path)
        return tmp_path

    # UI Handlers
    def set_preview_mode(self, mode):
        self.preview_mode = mode
        return self._apply_transform_inc(None)

    # Mirroring Buttons
    def mirror_x(self): return self._apply_transform_inc(np.diag([-1, 1, 1]))
    def mirror_y(self): return self._apply_transform_inc(np.diag([1, -1, 1]))
    def mirror_z(self): return self._apply_transform_inc(np.diag([1, 1, -1]))
    
    # Swapping Buttons
    def swap_xy(self):
        m = np.eye(3)
        m[0,0], m[0,1], m[1,0], m[1,1] = 0, 1, 1, 0
        return self._apply_transform_inc(m)
    def swap_xz(self):
        m = np.eye(3)
        m[0,0], m[0,2], m[2,0], m[2,2] = 0, 1, 1, 0
        return self._apply_transform_inc(m)
    def swap_yz(self):
        m = np.eye(3)
        m[1,1], m[1,2], m[2,1], m[2,2] = 0, 1, 1, 0
        return self._apply_transform_inc(m)

    # Rotation Buttons
    def rot_x_90(self): return self._apply_transform_inc(R.from_euler('x', 90, degrees=True).as_matrix())
    def rot_y_90(self): return self._apply_transform_inc(R.from_euler('y', 90, degrees=True).as_matrix())
    def rot_z_90(self): return self._apply_transform_inc(R.from_euler('z', 90, degrees=True).as_matrix())

    def save_asset(self):
        print(f"Applying final matrix transform to Gaussians:\n{self.current_matrix}")
        process_and_save_gaussians(
            folder_path=self.folder_path,
            transform_mat=self.current_matrix,
            t_xyz=np.array([0.0, 0.0, 0.0]),
            scaling_factor=self.scaling_factor,
            spotiness_correction_factor=self.spotiness_correction_factor
        )
        return "Asset saved successfully!"

    def run(self):
        with gr.Blocks(title="Asset Orientation Validation") as demo:
            gr.Markdown("## Validate Asset Orientation")
            
            with gr.Row():
                with gr.Column(scale=3):
                    model_viewer = gr.Model3D(value=self._generate_preview_glb(), clear_color=[0.0, 0.0, 0.0, 0.0])
                
                with gr.Column(scale=1):
                    mode_radio = gr.Radio(["Gaussians", "Mesh"], label="Preview Mode", value=self.preview_mode)
                    
                    gr.Markdown("### Orientation & Scale")
                    gr.Markdown("**Axis Guide (5m long):**\n- <span style='color:red'>Red</span>: +X (Front)\n- <span style='color:green'>Green</span>: +Y (Left)\n- <span style='color:blue'>Blue</span>: +Z (Up)")
                    
                    gr.Markdown("#### Mirroring (Reflections)")
                    with gr.Row():
                        btn_mirror_x = gr.Button("Mirror X")
                        btn_mirror_y = gr.Button("Mirror Y")
                        btn_mirror_z = gr.Button("Mirror Z")
                        
                    gr.Markdown("#### Axis Swapping")
                    with gr.Row():
                        btn_swap_xy = gr.Button("Swap X/Y")
                        btn_swap_xz = gr.Button("Swap X/Z")
                        btn_swap_yz = gr.Button("Swap Y/Z")
                        
                    gr.Markdown("#### Rotate 90°")
                    with gr.Row():
                        btn_rot_x_90 = gr.Button("Rot X")
                        btn_rot_y_90 = gr.Button("Rot Y")
                        btn_rot_z_90 = gr.Button("Rot Z")

                    gr.Markdown("---")
                    btn_save = gr.Button("Save Asset", variant="primary")
                    save_status = gr.Markdown("")
            
            mode_radio.change(fn=self.set_preview_mode, inputs=mode_radio, outputs=model_viewer)
            
            btn_mirror_x.click(fn=self.mirror_x, outputs=model_viewer)
            btn_mirror_y.click(fn=self.mirror_y, outputs=model_viewer)
            btn_mirror_z.click(fn=self.mirror_z, outputs=model_viewer)
            
            btn_swap_xy.click(fn=self.swap_xy, outputs=model_viewer)
            btn_swap_xz.click(fn=self.swap_xz, outputs=model_viewer)
            btn_swap_yz.click(fn=self.swap_yz, outputs=model_viewer)
            
            btn_rot_x_90.click(fn=self.rot_x_90, outputs=model_viewer)
            btn_rot_y_90.click(fn=self.rot_y_90, outputs=model_viewer)
            btn_rot_z_90.click(fn=self.rot_z_90, outputs=model_viewer)
            
            btn_save.click(fn=self.save_asset, outputs=save_status)
            
        print("Launching Gradio interface...")
        demo.launch(inline=True, share=True)

def start_validation(folder_path, spotiness_correction_factor):
    app = ValidationApp(folder_path, spotiness_correction_factor)
    app.run()
