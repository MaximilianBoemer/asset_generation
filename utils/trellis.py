import os
import glob
import shutil
import tempfile
from PIL import Image
from gradio_client import Client, handle_file

client = Client("microsoft/TRELLIS")

TIMEOUT = 300
MAX_MULTIVIEW_IMAGES = 4  # TRELLIS works best with 2-4 views; cap to avoid OOM

def prepare_image(path: str) -> str:
    """Resize and save image to a temp PNG, return the temp path."""
    img = Image.open(path).convert("RGB")
    img = img.resize((512, 512))
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    img.save(tmp.name)
    return tmp.name

def run_trellis_pipeline(image_folder: str, output_dir: str, multiview: bool = True):
    os.makedirs(output_dir, exist_ok=True)
    image_paths = sorted(glob.glob(os.path.join(image_folder, "*.png")))
    if not image_paths:
        raise FileNotFoundError(f"No PNG files found in {image_folder}")

    print(f"Found {len(image_paths)} images — multiview={multiview}")

    print("Starting session...")
    client.predict(api_name="/start_session")

    prepared_paths = [prepare_image(p) for p in image_paths]

    if multiview and len(prepared_paths) > 1:
        view_paths = prepared_paths[:MAX_MULTIVIEW_IMAGES]
        print(f"Preprocessing {len(view_paths)} views...")
        gallery_input = [
            {"image": handle_file(p), "caption": None}
            for p in view_paths
        ]
        preprocessed_gallery = client.predict(
            images=gallery_input,
            api_name="/preprocess_images"
        )
        print(f"Preprocessed gallery: {preprocessed_gallery}")

        org_view_paths = image_paths[:MAX_MULTIVIEW_IMAGES]
        max_res = 0
        largest_idx = 0
        for i, p in enumerate(org_view_paths):
            with Image.open(p) as img:
                res = img.width * img.height
                if res > max_res:
                    max_res = res
                    largest_idx = i

        primary_image = handle_file(preprocessed_gallery[largest_idx]["image"])
        multiimages = [
            {"image": handle_file(item["image"]), "caption": item.get("caption")}
            for item in preprocessed_gallery
        ]
        multiimage_algo = "multidiffusion"
    else:
        print("Preprocessing single image...")
        preprocessed = client.predict(
            image=handle_file(prepared_paths[0]),
            api_name="/preprocess_image"
        )
        print(f"Preprocessed image: {preprocessed}")

        primary_image = handle_file(preprocessed)
        multiimages = []
        multiimage_algo = "stochastic"

    seed = client.predict(
        randomize_seed=True,
        seed=0,
        api_name="/get_seed"
    )
    print(f"Using seed: {seed}")

    print("Submitting 3D generation job...")
    job = client.submit(
        image=primary_image,
        multiimages=multiimages,
        seed=seed,
        ss_guidance_strength=7.5,
        ss_sampling_steps=12,
        slat_guidance_strength=3.0,
        slat_sampling_steps=12,
        multiimage_algo=multiimage_algo,
        api_name="/image_to_3d"
    )

    res = job.result(timeout=TIMEOUT)
    print(f"3D generation finished. Result: {res}")

    print("Extracting GLB...")
    glb_res = client.submit(
        mesh_simplify=0.95,
        texture_size=1024,
        api_name="/extract_glb"
    ).result(timeout=TIMEOUT)

    print(f"GLB result: {glb_res}")
    glb_path = glb_res[1]
    target_glb = os.path.join(output_dir, "extracted_mesh.glb")
    shutil.copy(glb_path, target_glb)
    print(f"GLB saved to {target_glb}")

    print("Extracting Gaussians...")
    gs_res = client.submit(
        api_name="/extract_gaussian"
    ).result(timeout=TIMEOUT)

    print(f"Gaussian result: {gs_res}")
    gs_path = gs_res[1]
    target_gs = os.path.join(output_dir, "extracted_gaussians.ply")
    shutil.copy(gs_path, target_gs)
    print(f"Gaussians saved to {target_gs}")
