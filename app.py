# MIT License

# Copyright (c) Microsoft

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Copyright (c) [2025] [Microsoft]
# Copyright (c) [2025] [Chongjie Ye] 
# SPDX-License-Identifier: MIT
# This file has been modified by Chongjie Ye on 2025/04/10
# Original file was released under MIT, with the full license text # available at https://github.com/atong01/conditional-flow-matching/blob/1.0.7/LICENSE.
# This modified file is released under the same license.

import gradio as gr
import os
os.environ['SPCONV_ALGO'] = 'native'
from typing import *
import torch
import numpy as np
from hi3dgen.pipelines import Hi3DGenPipeline
import trimesh
import tempfile
import hf_transfer

MAX_SEED = np.iinfo(np.int32).max
TMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tmp')
WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weights')
os.makedirs(TMP_DIR, exist_ok=True)
os.makedirs(WEIGHTS_DIR, exist_ok=True)

# Initialize placeholders for models
hi3dgen_pipeline = None
normal_predictor = None


def cache_weights(weights_dir: str) -> dict:
    import os
    import sys
    from huggingface_hub import list_repo_files, hf_hub_download
    from pathlib import Path

    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1" # Attempt to use hf-transfer

    model_ids = [ # Renamed for brevity
        "Stable-X/trellis-normal-v0-1", "Stable-X/yoso-normal-v1-8-1", "ZhengPeng7/BiRefNet"
    ]
        print(f"Downloading and caching model: {model_id}")
        # Use snapshot_download to download the model
        local_path = snapshot_download(repo_id=model_id, local_dir=os.path.join(weights_dir, model_id.split("/")[-1]), force_download=False)
        cached_paths[model_id] = local_path
        print(f"Cached at: {local_path}")
                # Print progress before each file operation
                print(f"  [{i+1}/{num_total_files}] Processing: {filename_in_repo}")
                sys.stdout.flush()
                try:
                    # hf_hub_download handles caching internally.
                    # It downloads to a shared HF cache then copies/symlinks to local_dir if specified.
                    # We specify local_dir to ensure files land directly in our target structure.
                        repo_id=repo_id,
                    # More robust error handling would log this or retry. For compactness, we skip.
                    # print(f"    Warning: Skipping {filename_in_repo} due to: {str(file_e)[:50]}...") # Optional: short error
                    pass
            
            cached_paths[repo_id] = str(local_repo_root)
            print(f"  Finished processing {repo_id}.")
        except Exception as repo_e:
            print(f"  ERROR processing repository {repo_id}: {repo_e}")
        sys.stdout.flush()

    return cached_paths



def preprocess_mesh(mesh_prompt):
    print("Processing mesh")
    trimesh_mesh = trimesh.load_mesh(mesh_prompt)
    trimesh_mesh.export(mesh_prompt+'.glb')
    return mesh_prompt+'.glb'


def preprocess_image(image):
    global hi3dgen_pipeline
    if image is None: return None
    # hi3dgen_pipeline is loaded on CPU at startup. Critical if None.
    if hi3dgen_pipeline is None: 
        raise RuntimeError("FATAL: Hi3DGenPipeline not loaded. Cannot preprocess.")
    # .preprocess_image is expected to work with the pipeline on CPU
    # and manage its own sub-component (e.g., BiRefNet) devices.
    return hi3dgen_pipeline.preprocess_image(image, resolution=1024)


def generate_3d(image, seed=-1,  
                ss_guidance_strength=3, ss_sampling_steps=50,
                slat_guidance_strength=3, slat_sampling_steps=6,):
    global hi3dgen_pipeline, normal_predictor # Manage global model states

    if image is None: return None, None, None
    if seed == -1: seed = np.random.randint(0, MAX_SEED)
    
    if hi3dgen_pipeline is None:
        raise RuntimeError("FATAL: Hi3DGenPipeline not loaded. Cannot generate 3D.")

    normal_image_pil = None
    mesh_path_output = None
    
    # --- STAGE 1: Normal Prediction ---
    # Predictor class automatically moves its internal YOSONormalsPipeline to GPU if available during init.
    try:
        print("Normal Prediction: Loading model...") # It will auto-select GPU if available
        # local_cache_dir ensures hub model CODE is sought/placed in WEIGHTS_DIR/hugoycj_StableNormal_main
        # The Predictor internally calls from_pretrained for its model, using WEIGHTS_DIR as cache_dir for WEIGHTS.
        normal_predictor = torch.hub.load(
            "hugoycj/StableNormal", "StableNormal_turbo", trust_repo=True, 
            yoso_version='yoso-normal-v1-8-1', local_cache_dir=WEIGHTS_DIR
        )
        # NO explicit .cuda() call on normal_predictor (the wrapper object)
        
        print("Normal Prediction: Generating normal map...")
        normal_image_pil = normal_predictor(image, resolution=768, match_input_resolution=True, data_type='object')
    except Exception as e:
        print(f"ERROR in Normal Prediction stage: {e}")
        # Ensure normal_predictor is cleaned up even if prediction fails after loading
        if normal_predictor is not None:
            print("Normal Prediction: Unloading model (due to error)...")
            if hasattr(normal_predictor, 'model') and hasattr(normal_predictor.model, 'cpu'):
                normal_predictor.model.cpu() # Move internal model to CPU
            del normal_predictor; normal_predictor = None 
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        return None, None, None # Abort if normal prediction fails
    finally:
        # This block executes if try was successful (normal_image_pil is not None)
        # or if an exception occurred *not* handled by the inner except block above (less likely here).
        # If normal_image_pil is None here, it means an error happened and was handled, so we skip further cleanup.
        if normal_image_pil is not None and normal_predictor is not None:
            print("Normal Prediction: Unloading model...")
            if hasattr(normal_predictor, 'model') and hasattr(normal_predictor.model, 'cpu'):
                normal_predictor.model.cpu() # Move internal model to CPU
            del normal_predictor; normal_predictor = None 
            if torch.cuda.is_available(): torch.cuda.empty_cache()

    if normal_image_pil is None: # Check again, just in case (e.g. if finally didn't run as expected)
        print("ERROR: Normal map not generated. Aborting 3D generation.")
        return None, None, None

    # --- STAGE 2: 3D Generation ---
    # Moves hi3dgen_pipeline (on CPU) to GPU, uses it, then moves back to CPU.
    pipeline_on_gpu = False
    try:
        if torch.cuda.is_available():
            print("3D Generation: Moving pipeline to GPU...")
            hi3dgen_pipeline.cuda(); pipeline_on_gpu = True
        
        print("3D Generation: Running pipeline...")
        outputs = hi3dgen_pipeline.run(
            normal_image_pil, seed=seed, formats=["mesh",], preprocess_image=False,
            sparse_structure_sampler_params={"steps": ss_sampling_steps, "cfg_strength": ss_guidance_strength},
            slat_sampler_params={"steps": slat_sampling_steps, "cfg_strength": slat_guidance_strength},
        )
        
        import datetime 
        mesh_path_output = f"{TMP_DIR}/{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}/mesh.glb"
        os.makedirs(os.path.dirname(mesh_path_output), exist_ok=True)
        outputs['mesh'][0].to_trimesh(transform_pose=True).export(mesh_path_output)
        print(f"SUCCESS: Mesh exported to {mesh_path_output}")

    except Exception as e:
        print(f"ERROR in 3D Generation stage: {e}")
    finally:
        if pipeline_on_gpu: 
            print("3D Generation: Moving pipeline to CPU...")
            hi3dgen_pipeline.cpu()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            
    return normal_image_pil, mesh_path_output, mesh_path_output


def convert_mesh(mesh_path, export_format):
    """Download the mesh in the selected format."""
    if not mesh_path:
        return None
    
    # Create a temporary file to store the mesh data
    temp_file = tempfile.NamedTemporaryFile(suffix=f".{export_format}", delete=False)
    temp_file_path = temp_file.name
    
    new_mesh_path = mesh_path.replace(".glb", f".{export_format}")
    mesh = trimesh.load_mesh(mesh_path)
    mesh.export(temp_file_path)  # Export to the temporary file
    
    return temp_file_path # Return the path to the temporary file

# Create the Gradio interface with improved layout
with gr.Blocks(css="footer {visibility: hidden}") as demo:
    gr.Markdown(
        """
        <h1 style='text-align: center;'>Hi3DGen: High-fidelity 3D Geometry Generation from Images via Normal Bridging</h1>
        <p style='text-align: center;'>
            <strong>V0.1, Introduced By 
            <a href="https://gaplab.cuhk.edu.cn/" target="_blank">GAP Lab</a> from CUHKSZ and 
            <a href="https://www.nvsgames.cn/" target="_blank">Game-AIGC Team</a> from ByteDance</strong>
        </p>
        """
    )
    
    with gr.Row():
        gr.Markdown("""
                    <p align="center">
                    <a title="Website" href="https://stable-x.github.io/Hi3DGen/" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
                        <img src="https://www.obukhov.ai/img/badges/badge-website.svg">
                    </a>
                    <a title="arXiv" href="https://stable-x.github.io/Hi3DGen/hi3dgen_paper.pdf" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
                        <img src="https://www.obukhov.ai/img/badges/badge-pdf.svg">
                    </a>
                    <a title="Github" href="https://github.com/Stable-X/Hi3DGen" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
                        <img src="https://img.shields.io/github/stars/Stable-X/Hi3DGen?label=GitHub%20%E2%98%85&logo=github&color=C8C" alt="badge-github-stars">
                    </a>
                    <a title="Social" href="https://x.com/ychngji6" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
                        <img src="https://www.obukhov.ai/img/badges/badge-social.svg" alt="social">
                    </a>
                    </p>
                    """)

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Tabs():
                
                with gr.Tab("Single Image"):
                    with gr.Row():
                        image_prompt = gr.Image(label="Image Prompt", image_mode="RGBA", type="pil")
                        normal_output = gr.Image(label="Normal Bridge", image_mode="RGBA", type="pil")
                        
                with gr.Tab("Multiple Images"):
                    gr.Markdown("<div style='text-align: center; padding: 40px; font-size: 24px;'>Multiple Images functionality is coming soon!</div>")
                        
            with gr.Accordion("Advanced Settings", open=False):
                seed = gr.Slider(-1, MAX_SEED, label="Seed", value=0, step=1)
                gr.Markdown("#### Stage 1: Sparse Structure Generation")
                with gr.Row():
                    ss_guidance_strength = gr.Slider(0.0, 10.0, label="Guidance Strength", value=3, step=0.1)
                    ss_sampling_steps = gr.Slider(1, 50, label="Sampling Steps", value=50, step=1)
                gr.Markdown("#### Stage 2: Structured Latent Generation")
                with gr.Row():
                    slat_guidance_strength = gr.Slider(0.0, 10.0, label="Guidance Strength", value=3.0, step=0.1)
                    slat_sampling_steps = gr.Slider(1, 50, label="Sampling Steps", value=6, step=1)
                    
            with gr.Group():
                with gr.Row():
                    gen_shape_btn = gr.Button("Generate Shape", size="lg", variant="primary")
                        
        # Right column - Output
        with gr.Column(scale=1):
            with gr.Column():
                model_output = gr.Model3D(label="3D Model Preview (Each model is approximately 40MB, may take around 1 minute to load)")
            with gr.Column():
                export_format = gr.Dropdown(
                    choices=["obj", "glb", "ply", "stl"],
                    value="glb",
                    label="File Format"
                )
                download_btn = gr.DownloadButton(label="Export Mesh", interactive=False)

    image_prompt.upload(
        preprocess_image,
        inputs=[image_prompt],
        outputs=[image_prompt]
    )
    
    gen_shape_btn.click(
        generate_3d,
        inputs=[
            image_prompt, seed,  
            ss_guidance_strength, ss_sampling_steps,
            slat_guidance_strength, slat_sampling_steps
        ],
        outputs=[normal_output, model_output, download_btn]
    ).then(
        lambda: gr.Button(interactive=True),
        outputs=[download_btn],
    )
    
    
    def update_download_button(mesh_path, export_format):
        if not mesh_path:
            return gr.File.update(value=None, interactive=False)
        
        download_path = convert_mesh(mesh_path, export_format)
        return download_path
    
    export_format.change(
        update_download_button,
        inputs=[model_output, export_format],
        outputs=[download_btn]
    ).then(
        lambda: gr.Button(interactive=True),
        outputs=[download_btn],
    )
    
    examples = gr.Examples(
        examples=[
            f'assets/example_image/{image}'
            for image in os.listdir("assets/example_image")
        ],
        inputs=image_prompt,
    )

    gr.Markdown(
        """
        **Acknowledgments**: Hi3DGen is built on the shoulders of giants. We would like to express our gratitude to the open-source research community and the developers of these pioneering projects:
        - **3D Modeling:** Our 3D Model is finetuned from the SOTA open-source 3D foundation model [Trellis](https://github.com/microsoft/TRELLIS) and we draw inspiration from the teams behind [Rodin](https://hyperhuman.deemos.com/rodin), [Tripo](https://www.tripo3d.ai/app/home), and [Dora](https://github.com/Seed3D/Dora).
        - **Normal Estimation:** Our Normal Estimation Model builds on the leading normal estimation research such as [StableNormal](https://github.com/hugoycj/StableNormal) and [GenPercept](https://github.com/aim-uofa/GenPercept).
        
        **Your contributions and collaboration push the boundaries of 3D modeling!**
        """
    )

if __name__ == "__main__":
    # Download and cache the weights
    cache_weights(WEIGHTS_DIR)

    print("Loading Hi3DGenPipeline to CPU...")
    hi3dgen_pipeline = Hi3DGenPipeline.from_pretrained("weights/trellis-normal-v0-1")
    hi3dgen_pipeline.cpu() # Move to CPU after loading
    print("Hi3DGenPipeline loaded on CPU.")

    # normal_predictor will be loaded on demand in generate_3d function
    
    # Launch the app
    demo.launch(share=False, server_name="127.0.0.1")

