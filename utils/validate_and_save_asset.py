import gradio as gr
import trimesh
import os
import sys
import json
import numpy as np
from scipy.spatial.transform import Rotation as R

from utils.gaussian_utils import process_and_save_gaussians

def guess_initial_transform(mesh_path, dims_json_path):
    if not os.path.exists(dims_json_path):
        raise FileNotFoundError(f"dims.json not found at {dims_json_path}")
        
    with open(dims_json_path, 'r') as f:
        target_dims = json.load(f)
        
    mesh = trimesh.load(mesh_path, process=False)
    bounds = mesh.bounds
    orig_dims_vec = (bounds[1] - bounds[0]) # dx, dy, dz
    
    # Map largest to length (X), middle to width (Y), smallest to height (Z)
    orig_order = np.argsort(orig_dims_vec)[::-1] 
    
    mat = np.zeros((3, 3))
    mat[0, orig_order[0]] = 1.0 # New X axis
    mat[1, orig_order[1]] = 1.0 # New Y axis
    mat[2, orig_order[2]] = 1.0 # New Z axis
    
    # We allow reflections in the initial guess if it matches the axes best
    scaling_factor = target_dims['length'] / orig_dims_vec[orig_order[0]]
    print(f"Using guessed initial matrix:\n{mat}\nand scaling_factor: {scaling_factor}")
    return mat, scaling_factor

class ValidationApp:
    def __init__(self, folder_path, spotiness_correction_factor):
        self.folder_path = folder_path
        self.spotiness_correction_factor = spotiness_correction_factor
        
        self.base_glb_path = os.path.join(folder_path, "extracted_mesh.glb")
        self.dims_json_path = os.path.join(os.path.dirname(folder_path), "dims.json")
        
        self.current_matrix, self.scaling_factor = guess_initial_transform(self.base_glb_path, self.dims_json_path)
        
    def _apply_transform_inc(self, inc_matrix):
        """Apply an incremental 3x3 matrix to the current transformation (extrinsic/world-space view)"""
        self.current_matrix = inc_matrix @ self.current_matrix
        return self._generate_preview_glb()

    def _generate_preview_glb(self):
        """Generates a preview glb with matrix transform and world axis/grid helper"""
        mesh = trimesh.load(self.base_glb_path, process=False)
        
        # Apply transformation and scale
        transform = np.eye(4)
        transform[:3, :3] = self.current_matrix * self.scaling_factor
        mesh.apply_transform(transform)
        
        # Centering (match process_and_save_gaussians logic)
        bounds = mesh.bounds
        t_xyz_center = -0.5 * (bounds[0] + bounds[1])
        mesh.apply_translation(t_xyz_center)
        
        # 5m axis helper (Red=X, Green=Y, Blue=Z)
        axis_mesh = trimesh.creation.axis(origin_size=0.05, axis_radius=0.02, axis_length=5.0)
        
        # Grid helper (10m x 10m, 1m intervals)
        grid_lines = []
        for i in range(-5, 6):
            grid_lines.append([[i, -5, 0], [i, 5, 0]])
            grid_lines.append([[-5, i, 0], [5, i, 0]])
        grid = trimesh.load_path(grid_lines)
        
        scene = trimesh.Scene([mesh, axis_mesh, grid])
        
        tmp_path = os.path.join(self.folder_path, "preview_mesh.glb")
        scene.export(tmp_path)
        return tmp_path

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

    # 90-deg Rotations (Utility)
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
            gr.Markdown("## Validate Asset Orientation (Matrix-based)")
            
            with gr.Row():
                with gr.Column(scale=3):
                    model_viewer = gr.Model3D(value=self._generate_preview_glb(), clear_color=[0.0, 0.0, 0.0, 0.0])
                
                with gr.Column(scale=1):
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
                        btn_invert_all = gr.Button("Invert All", variant="secondary")
                        
                    gr.Markdown("#### Rotate 90°")
                    with gr.Row():
                        btn_rot_x_90 = gr.Button("Rot X")
                        btn_rot_y_90 = gr.Button("Rot Y")
                        btn_rot_z_90 = gr.Button("Rot Z")

                    gr.Markdown("---")
                    btn_save = gr.Button("Save Asset", variant="primary")
                    save_status = gr.Markdown("")
            
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
