import gradio as gr
import pandas as pd
import os
from multiprocessing import JoinableQueue
from PIL import Image
from oneshot_train import Trainer
import base64
from pathlib import Path


def check_and_upload(current_images):
    if current_images is not None and len(current_images) > 0:
        gr.Warning("Only one image can be uploaded. To change the image, please click 'Clear Photos' first.")
    return current_images

def show_queue_csv_path():
    # Use Path object to handle paths
    current_path = Path(__file__).parent.joinpath('job_status.csv')
    gr.Info(f"History jobs are saved in {current_path}", duration=20)

# Main interface building function
def train_input(queue, job_status_manager):
    trainer = Trainer(queue, job_status_manager)
    
    with gr.Blocks(css="frame_train.css") as demo:
        with gr.Row(): 
            with gr.Column(scale=1):
                with gr.Column():
                    output_model_name = gr.Textbox(
                        label="Lora Model Name", 
                        value='person1',
                        lines=1
                    )
                    instance_images = gr.Gallery(
                        label="Image Preview",
                        show_label=True,
                        elem_classes="red-background"
                    )

                    trigger_words = gr.Textbox(  
                        label="Trigger word/sentence",
                        info="Trigger word or sentence to be used",
                        value="b3tty001",
                        placeholder="uncommon word like b3tty001 or trtcrd, or sentence like 'in the style of CNSTLL'",
                        interactive=True
                    )

                    image_caption = gr.Textbox(
                        label="Image Caption", 
                        value="[trigger] a photo of a person",
                        lines=1
                    )
                with gr.Row():
                    upload_button = gr.UploadButton(
                        "Upload Photo",
                            file_types=["image"],
                    )
                    clear_button = gr.Button("Clear Photos")

                    # Add click event for upload button
                    upload_button.upload(
                        fn=check_and_upload, inputs=[instance_images],
                        outputs=instance_images
                    )
                    clear_button.click(
                        fn=lambda: None,
                            inputs=[],
                        outputs=instance_images
                    )
            
            with gr.Column(scale=1):
                with gr.Row(elem_classes="transparent-group"):
                    # table header
                    job_queue = gr.Dataframe(
                        headers=["Image","Job ID", "Status", "Completion Time", "Caption", "Model Name"],
                        datatype=["html", "str", "str", "str", "str", "str"],
                        value=[],
                        interactive=False,
                        row_count=10,
                        max_height="500",
                        elem_classes="job-queue"
                    )
                    
                with gr.Row():
                    refresh_button = gr.Button("Refresh Task Status")
                    history_button = gr.Button("History Jobs")

        with gr.Row():
            with gr.Column():
                gr.Markdown("""
                ### Instructions:
                1. Upload your ID photo
                2. Click [Start Training] button to customize your digital avatar (each Flux takes about 1.5 hours)
                3. Switch to [Generate] tab to generate photos
                """)
                
                run_button = gr.Button(
                    'Start Training (Please wait for photos to upload completely, otherwise training may fail)',
                    variant="primary"
                )

        run_button.click(
            # fn=lambda model_name, images, prompt: (trainer.run(model_name, images, prompt), []),
            fn=trainer.run,
            inputs=[output_model_name, instance_images, trigger_words, image_caption],
            outputs=[instance_images, job_queue]
        )

        # Change change event to button click event
        refresh_button.click(fn=trainer.job_manager.get_all_records, inputs=[], outputs=[job_queue])

        # Change history_button click event
        history_button.click(
            fn=show_queue_csv_path
        )

        # Add load event to update DataFrame data when the page loads
        demo.load(
            fn=trainer.job_manager.get_all_records,
            inputs=[],
            outputs=[job_queue]
        )

    return demo
