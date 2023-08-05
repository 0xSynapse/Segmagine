import os

import cv2
import gradio as gr
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from PIL import Image

from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry

# suppress server-side GUI windows
matplotlib.pyplot.switch_backend('Agg') 

# setup models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sam = sam_model_registry["vit_b"](checkpoint="./sam_vit_b_01ec64.pth")
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(sam)
predictor = SamPredictor(sam)


# copied from: https://github.com/facebookresearch/segment-anything
def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    withContours=True
    polygons = []
    color = []
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.35)))


# demo function
def segment_image(input_image):

    if input_image is not None:

        # generate masks
        masks = mask_generator.generate(input_image)

        # add masks to image
        plt.clf()
        ppi = 100
        height, width, _ = input_image.shape
        plt.figure(figsize=(width / ppi, height / ppi))  # convert pixel to inches
        plt.imshow(input_image)
        show_anns(masks)
        plt.axis('off')

        # save and get figure
        plt.savefig('output_figure.png', bbox_inches='tight')
        output_image = cv2.imread('output_figure.png')
        return Image.fromarray(output_image)


with gr.Blocks() as demo:

    with gr.Row():
        gr.Markdown("## Segment Anything (by Meta AI Research)")
    with gr.Row():
        gr.Markdown("The Segment Anything Model (SAM) produces high quality object masks from input prompts such as points or boxes, and it can be used to generate masks for all objects in an image. It has been trained on a dataset of 11 million images and 1.1 billion masks, and has strong zero-shot performance on a variety of segmentation tasks.")

    with gr.Row():

        with gr.Column():
            image_input = gr.Image()
            segment_image_button = gr.Button('Generate Mask')

        with gr.Column():
            image_output = gr.Image()

    segment_image_button.click(segment_image, inputs=[image_input], outputs=image_output)

    gr.Examples(
        examples=[
            ['./examples/dog.jpg'],
            ['./examples/groceries.jpg'],
            ['./examples/truck.jpg']

        ],
        inputs=[image_input],
        outputs=[image_output],
        fn=segment_image,
        #cache_examples=True
    )

demo.launch()
