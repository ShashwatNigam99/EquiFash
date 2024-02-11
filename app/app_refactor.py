import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
os.chdir("../")
import cv2
import gradio as gr
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
import torch
import tempfile
from stable_diffusion_inpaint import replace_img_with_sd, fill_img_with_sd
from lama_inpaint import inpaint_img_with_lama, build_lama_model, inpaint_img_with_builded_lama
from utils import load_img_to_array, save_array_to_img, dilate_mask, \
    show_mask, show_points
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry
import argparse
import copy

def setup_args(parser):
    parser.add_argument(
        "--lama_config", type=str,
        default="./lama/configs/prediction/default.yaml",
        help="The path to the config file of lama model. "
             "Default: the config of big-lama",
    )
    parser.add_argument(
        "--lama_ckpt", type=str,
        default="pretrained_models/big-lama",
        help="The path to the lama checkpoint.",
    )
    parser.add_argument(
        "--sam_ckpt", type=str,
        default="./pretrained_models/sam_vit_h_4b8939.pth",
        help="The path to the SAM checkpoint to use for mask generation.",
    )

def mkstemp(suffix, dir=None):
    fd, path = tempfile.mkstemp(suffix=f"{suffix}", dir=dir)
    os.close(fd)
    return Path(path)

def get_sam_feat(img):
    model['sam'].set_image(img)
    features = model['sam'].features 
    orig_h = model['sam'].orig_h 
    orig_w = model['sam'].orig_w 
    input_h = model['sam'].input_h 
    input_w = model['sam'].input_w 
    model['sam'].reset_image()
    return features, orig_h, orig_w, input_h, input_w

def get_replace_img_with_sd(image, mask, image_resolution, text_prompt):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if len(mask.shape)==3:
        mask = mask[:,:,0]
    np_image = np.array(image, dtype=np.uint8)
    H, W, C = np_image.shape
    np_image = HWC3(np_image)
    np_image = resize_image(np_image, image_resolution)

    img_replaced = replace_img_with_sd(np_image, mask, text_prompt, device=device)
    img_replaced = img_replaced.astype(np.uint8)
    return img_replaced

def get_filled_img_with_sd(image, mask, image_resolution, text_prompt):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if len(mask.shape)==3:
        mask = mask[:,:,0]
    np_image = np.array(image, dtype=np.uint8)
    H, W, C = np_image.shape
    np_image = HWC3(np_image)
    np_image = resize_image(np_image, image_resolution)

    img_filled = fill_img_with_sd(np_image, mask, text_prompt, device=device)
    img_filled = img_filled.astype(np.uint8)
    return img_filled

def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y

def resize_image(input_image, resolution):
    H, W, C = input_image.shape
    H = float(H)
    W = float(W)
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(np.round(H / 64.0)) * 64
    W = int(np.round(W / 64.0)) * 64
    img = cv2.resize(input_image, (W, H), interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)
    return img

def resize_points(clicked_points, original_shape, resolution):
    original_height, original_width, _ = original_shape
    original_height = float(original_height)
    original_width = float(original_width)
    
    scale_factor = float(resolution) / min(original_height, original_width)
    resized_points = []
    
    for point in clicked_points:
        x, y, lab = point
        resized_x = int(round(x * scale_factor))
        resized_y = int(round(y * scale_factor))
        resized_point = (resized_x, resized_y, lab)
        resized_points.append(resized_point)
    
    return resized_points

def get_click_mask(clicked_points, features, orig_h, orig_w, input_h, input_w, dilate_kernel_size):
    # model['sam'].set_image(image)
    model['sam'].is_image_set = True
    model['sam'].features = features
    model['sam'].orig_h = orig_h
    model['sam'].orig_w = orig_w
    model['sam'].input_h = input_h
    model['sam'].input_w = input_w
    
    # Separate the points and labels
    points, labels = zip(*[(point[:2], point[2])
                            for point in clicked_points])

    # Convert the points and labels to numpy arrays
    input_point = np.array(points)
    input_label = np.array(labels)

    masks, _, _ = model['sam'].predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False,
    )
    if dilate_kernel_size is not None:
        masks = [dilate_mask(mask, dilate_kernel_size) for mask in masks]
    else:
        masks = [mask for mask in masks]

    return masks

def process_image_click(original_image, point_prompt, clicked_points, image_resolution, features, orig_h, orig_w, input_h, input_w, dilate_kernel_size, evt: gr.SelectData):
    clicked_coords = evt.index
    x, y = clicked_coords
    label = point_prompt
    lab = 1 if label == "Foreground Point" else 0
    clicked_points.append((x, y, lab))

    input_image = np.array(original_image, dtype=np.uint8)
    H, W, C = input_image.shape
    input_image = HWC3(input_image)
    img = resize_image(input_image, image_resolution)

    # Update the clicked_points
    resized_points = resize_points(
        clicked_points, input_image.shape, image_resolution
    )
    mask_click_np = get_click_mask(resized_points, features, orig_h, orig_w, input_h, input_w, dilate_kernel_size)

    # Convert mask_click_np to HWC format
    mask_click_np = np.transpose(mask_click_np, (1, 2, 0)) * 255.0

    mask_image = HWC3(mask_click_np.astype(np.uint8))
    mask_image = cv2.resize(
        mask_image, (W, H), interpolation=cv2.INTER_LINEAR)
    # mask_image = Image.fromarray(mask_image_tmp)

    # Draw circles for all clicked points
    edited_image = input_image
    for x, y, lab in clicked_points:
        # Set the circle color based on the label
        color = (255, 0, 0) if lab == 1 else (0, 0, 255)

        # Draw the circle
        edited_image = cv2.circle(edited_image, (x, y), 20, color, -1)

    # Set the opacity for the mask_image and edited_image
    opacity_mask = 0.75
    opacity_edited = 1.0

    # Combine the edited_image and the mask_image using cv2.addWeighted()
    overlay_image = cv2.addWeighted(
        edited_image,
        opacity_edited,
        (mask_image *
            np.array([0 / 255, 255 / 255, 0 / 255])).astype(np.uint8),
        opacity_mask,
        0,
    )

    return (
        overlay_image,
        # Image.fromarray(overlay_image),
        clicked_points,
        # Image.fromarray(mask_image),
        mask_image
    )

def image_upload(image, image_resolution):
    if image is not None:
        np_image = np.array(image, dtype=np.uint8)
        H, W, C = np_image.shape
        np_image = HWC3(np_image)
        np_image = resize_image(np_image, image_resolution)
        features, orig_h, orig_w, input_h, input_w = get_sam_feat(np_image)
        return image, features, orig_h, orig_w, input_h, input_w, [], None
    else:
        return None, None, None, None, None, None, [], None

def get_inpainted_img(image, mask, image_resolution):
    lama_config = args.lama_config
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if len(mask.shape)==3:
        mask = mask[:,:,0]
    img_inpainted = inpaint_img_with_builded_lama(
        model['lama'], image, mask, lama_config, device=device)
    return img_inpainted

def set_output_to_input_resetpoints(output_image, image_resolution):
    source_image_click = copy.deepcopy(output_image)
    input_image, features, orig_h, orig_w, input_h, input_w, clicked_points, click_mask = image_upload(output_image, image_resolution)
    return source_image_click, input_image, features, orig_h, orig_w, input_h, input_w, clicked_points, click_mask

def reset_sam_points_mask(origin_image, clicked_points, click_mask):
    source_image_click = copy.deepcopy(origin_image)
    return source_image_click, [], None

# get args 
parser = argparse.ArgumentParser()
setup_args(parser)
args = parser.parse_args(sys.argv[1:])
# build models
model = {}
# build the sam model
model_type="vit_h"
ckpt_p=args.sam_ckpt
model_sam = sam_model_registry[model_type](checkpoint=ckpt_p)
device = "cuda" if torch.cuda.is_available() else "cpu"
model_sam.to(device=device)
model['sam'] = SamPredictor(model_sam)

# build the lama model
lama_config = args.lama_config
lama_ckpt = args.lama_ckpt
device = "cuda" if torch.cuda.is_available() else "cpu"
model['lama'] = build_lama_model(lama_config, lama_ckpt, device=device)

button_size = (100,50)
with gr.Blocks() as demo:
    clicked_points = gr.State([])
    origin_image = gr.State(None)
    click_mask = gr.State(None)
    features = gr.State(None)
    orig_h = gr.State(None)
    orig_w = gr.State(None)
    input_h = gr.State(None)
    input_w = gr.State(None)

    with gr.Row():
        with gr.Column(variant="panel"):
            with gr.Row():
                gr.Markdown("## Input Image")
            with gr.Row():
                # img = gr.Image(label="Input Image")
                source_image_click = gr.Image(
                    type="numpy",
                    interactive=True,
                    label="Image: Upload an image and click the region you want to edit.",
                )
        with gr.Column(variant="panel"):
            with gr.Row():
                gr.Markdown("## Output Image")
            with gr.Row():
                # img = gr.Image(label="Input Image")
                output_image = gr.Image(
                    type="numpy",
                    label="Output Image",
                    show_download_button=True,
                    interactive=False
                )

    with gr.Row():
        with gr.Column(variant="panel"):
            with gr.Row():
                point_prompt = gr.Radio(
                    choices=["Foreground Point",
                                "Background Point"],
                    value="Foreground Point",
                    label="Point Label",
                    interactive=True,
                    show_label=False,
                )
                reset_sam = gr.Button(
                    value="Reset points", 
                    variant="secondary"
                )
                # update_mask = gr.Button(
                #     value="Update Mask",
                #     variant="secondary"
                # )
            with gr.Row():  
                image_resolution = gr.Slider(
                    label="Image Resolution",
                    minimum=256,
                    maximum=1200,
                    value=1024,
                    step=32,
                )
                dilate_kernel_size = gr.Slider(
                    label="Dilate Kernel Size", 
                    minimum=0, 
                    maximum=64, 
                    step=2, 
                    value=16
                )
        with gr.Column(variant="panel"):
            text_prompt = gr.Textbox(label="Text Prompt")
            with gr.Row():
                inpaint_sd = gr.Button("Replace masked area", variant="primary")
                lama = gr.Button("Remove it!", variant="primary")
                # this we will replace with replace everything else
                replace_sd = gr.Button("Replace everything else!", variant="primary")
            with gr.Row():
                clear_button_image = gr.Button(value="Reset", variant="secondary")
                send_output_to_input = gr.Button(value="Send Output to Input", variant="primary")

    # todo: maybe we can delete this row, for it's unnecessary to show the original mask for customers
    with gr.Accordion("Open for More!", open=False):
        with gr.Row(variant="panel"):
            with gr.Column():
                with gr.Row():
                    gr.Markdown("## Mask")
                with gr.Row():
                    click_mask = gr.Image(type="numpy", label="Click Mask")
            with gr.Column():
                with gr.Row():
                    gr.Markdown("## Image Removed with Mask")
                with gr.Row():
                    img_rm_with_mask = gr.Image(
                        type="numpy", label="Image Removed with Mask")


    source_image_click.upload(
        image_upload,
        inputs=[source_image_click, image_resolution],
        outputs=[origin_image, features, orig_h, orig_w, input_h, input_w, clicked_points, click_mask],
    )

    source_image_click.select(
        process_image_click,
        inputs=[origin_image, point_prompt,
                clicked_points, image_resolution,
                features, orig_h, orig_w, input_h, input_w, dilate_kernel_size],
        outputs=[source_image_click, clicked_points, click_mask],
        show_progress=True,
        queue=True,
    )

    send_output_to_input.click(
        set_output_to_input_resetpoints,
        inputs  = [output_image, image_resolution],
        outputs = [source_image_click, origin_image, features, orig_h, orig_w, input_h, input_w, clicked_points, click_mask]
    )

    reset_sam.click(
        reset_sam_points_mask,
        [origin_image, clicked_points, click_mask],
        [source_image_click, clicked_points, click_mask]
    )
    lama.click(
        get_inpainted_img,
        [origin_image, click_mask, image_resolution],
        [output_image]
    )
    
    replace_sd.click(
        get_replace_img_with_sd,
        [origin_image, click_mask, image_resolution, text_prompt],
        [output_image]
    )

    inpaint_sd.click(
        get_filled_img_with_sd,
        [origin_image, click_mask, image_resolution, text_prompt],
        [output_image]
    )


    def reset(*args):
        return [None for _ in args]

    clear_button_image.click(
        reset,
        [origin_image, features, click_mask, img_rm_with_mask, output_image],
        [origin_image, features, click_mask, img_rm_with_mask, output_image]
    )

if __name__ == "__main__":
    demo.queue(api_open=False).launch(server_name='0.0.0.0', share=True, debug=True)