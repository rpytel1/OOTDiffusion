import gradio as gr
import os
from pathlib import Path
import sys
import torch
from PIL import Image, ImageOps

from utils_ootd import get_mask_location

PROJECT_ROOT = Path(__file__).absolute().parents[1].absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from preprocess.openpose.run_openpose import OpenPose
from preprocess.humanparsing.run_parsing import Parsing
from ootd.inference_ootd_hd import OOTDiffusionHD


openpose_model_hd = OpenPose(0)
parsing_model_hd = Parsing(0)
ootd_model_hd = OOTDiffusionHD(0)

category_dict = ['upperbody', 'lowerbody', 'dress']
category_dict_utils = ['upper_body', 'lower_body', 'dresses']


example_path = os.path.join(os.path.dirname(__file__), 'examples')
model_hd = os.path.join(example_path, 'model/model_1.png')
garment_hd = os.path.join(example_path, 'garment/03244_00.jpg')
model_dc = os.path.join(example_path, 'model/model_8.png')
garment_dc = os.path.join(example_path, 'garment/048554_1.jpg')

def process_hd(vton_img, garm_img, n_samples, n_steps, image_scale, seed):
    model_type = 'hd'
    category = 0 # 0:upperbody; 1:lowerbody; 2:dress

    with torch.no_grad():
        garm_img = Image.open(garm_img).resize((768, 1024))
        vton_img = Image.open(vton_img).resize((768, 1024))
        keypoints = openpose_model_hd(vton_img.resize((384, 512)))
        model_parse, _ = parsing_model_hd(vton_img.resize((384, 512)))

        mask, mask_gray = get_mask_location(model_type, category_dict_utils[category], model_parse, keypoints)
        mask = mask.resize((768, 1024), Image.NEAREST)
        mask_gray = mask_gray.resize((768, 1024), Image.NEAREST)
        
        masked_vton_img = Image.composite(mask_gray, vton_img, mask)

        images = ootd_model_hd(
            model_type=model_type,
            category=category_dict[category],
            image_garm=garm_img,
            image_vton=masked_vton_img,
            mask=mask,
            image_ori=vton_img,
            num_samples=n_samples,
            num_steps=n_steps,
            image_scale=image_scale,
            seed=seed,
        )

    return images

block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown("# OOTDiffusion Demo")
    with gr.Row():
        gr.Markdown("## Half-body")
    with gr.Row():
        gr.Markdown("***Support upper-body garments***")
    with gr.Row():
        with gr.Column():
            vton_img = gr.Image(label="Model", sources='upload', type="filepath", height=384, value=model_hd)
            example = gr.Examples(
                inputs=vton_img,
                examples_per_page=14,
                examples=[
                    os.path.join(example_path, 'model/model_1.png'),
                    os.path.join(example_path, 'model/model_2.png'),
                    os.path.join(example_path, 'model/model_3.png'),
                    os.path.join(example_path, 'model/model_4.png'),
                    os.path.join(example_path, 'model/model_5.png'),
                    os.path.join(example_path, 'model/model_6.png'),
                    os.path.join(example_path, 'model/model_7.png'),
                    os.path.join(example_path, 'model/01008_00.jpg'),
                    os.path.join(example_path, 'model/07966_00.jpg'),
                    os.path.join(example_path, 'model/05997_00.jpg'),
                    os.path.join(example_path, 'model/02849_00.jpg'),
                    os.path.join(example_path, 'model/14627_00.jpg'),
                    os.path.join(example_path, 'model/09597_00.jpg'),
                    os.path.join(example_path, 'model/01861_00.jpg'),
                ])
        with gr.Column():
            garm_img = gr.Image(label="Garment", sources='upload', type="filepath", height=384, value=garment_hd)
            example = gr.Examples(
                inputs=garm_img,
                examples_per_page=14,
                examples=[
                    os.path.join(example_path, 'garment/03244_00.jpg'),
                    os.path.join(example_path, 'garment/00126_00.jpg'),
                    os.path.join(example_path, 'garment/03032_00.jpg'),
                    os.path.join(example_path, 'garment/06123_00.jpg'),
                    os.path.join(example_path, 'garment/02305_00.jpg'),
                    os.path.join(example_path, 'garment/00055_00.jpg'),
                    os.path.join(example_path, 'garment/00470_00.jpg'),
                    os.path.join(example_path, 'garment/02015_00.jpg'),
                    os.path.join(example_path, 'garment/10297_00.jpg'),
                    os.path.join(example_path, 'garment/07382_00.jpg'),
                    os.path.join(example_path, 'garment/07764_00.jpg'),
                    os.path.join(example_path, 'garment/00151_00.jpg'),
                    os.path.join(example_path, 'garment/12562_00.jpg'),
                    os.path.join(example_path, 'garment/04825_00.jpg'),
                ])
        with gr.Column():
            result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery", preview=True, scale=1)   
    with gr.Column():
        run_button = gr.Button(value="Run")
        n_samples = gr.Slider(label="Images", minimum=1, maximum=4, value=1, step=1)
        n_steps = gr.Slider(label="Steps", minimum=20, maximum=40, value=20, step=1)
        # scale = gr.Slider(label="Scale", minimum=1.0, maximum=12.0, value=5.0, step=0.1)
        image_scale = gr.Slider(label="Guidance scale", minimum=1.0, maximum=5.0, value=2.0, step=0.1)
        seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, value=-1)
        
    ips = [vton_img, garm_img, n_samples, n_steps, image_scale, seed]
    run_button.click(fn=process_hd, inputs=ips, outputs=[result_gallery])

block.launch(server_name='0.0.0.0', server_port=7865)
