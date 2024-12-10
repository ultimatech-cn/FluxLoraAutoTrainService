import gradio as gr
import pandas as pd
import os
from multiprocessing import JoinableQueue
from PIL import Image
from fewshot.fewshot_train import Trainer
import base64
from pathlib import Path

def get_done_jobs():
    job_df = pd.read_csv('job_status.csv')
    done_jobs = job_df['jobid'][job_df['status'] == 'done'][job_df['job_type'] == 'ONESHOT_GEN'].tolist()
    return done_jobs


# Main interface building function
def train_multi_input(queue, job_status_manager):
    trainer = Trainer(queue, job_status_manager)
    
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column(scale=1, min_width=300):
                done_jobs = get_done_jobs()
                job_selector = gr.Dropdown(
                    choices=done_jobs,
                    label="Select JobID of completed model from step 1",
                    multiselect=False,
                    value=None
                )   

                model_name = gr.Textbox(
                    label="Model Name",
                    interactive=False
                )

                caption = gr.Textbox(
                    label="Image Description",
                    interactive=False 
                )

                completion_time = gr.Textbox(
                    label="Completion Time",
                    interactive=False
                )

            with gr.Column(scale=1, min_width=300):
                preview_image = gr.Image(
                    label="Preview Image",
                    interactive=False,
                    type="filepath",
                    height=300
                )

        with gr.Accordion("Generated Images", open=False) as sample:
            generated_images = gr.Gallery(
                label=None, 
                show_label=False,
                columns=5,
                object_fit="contain",
                selected_index=0
            )

        with gr.Accordion("Selected Images", open=False) as selected_sample:
            selected_images = gr.Gallery(
                label=None,
                show_label=False,
                columns=5,
                object_fit="contain"
            )

        with gr.Row():
            run_button = gr.Button('Start Generation', variant="primary")
            select_button = gr.Button('Select Images')

        def update_display(selected_job):
            if selected_job:
                job_df = pd.read_csv("job_status.csv")
                job_info = job_df[job_df['jobid'] == selected_job].iloc[0]
                
                prefix = job_info['caption'].split(' ')[0] if job_info['caption'] else None
                
                generated_dir = f'job_data/{selected_job}/oneshot_generate'
                gallery_value = []
                if os.path.exists(generated_dir):
                    gallery_value = [
                        os.path.join(generated_dir, img) 
                        for img in os.listdir(generated_dir) 
                        if img.lower().endswith(('.png', '.jpg', '.jpeg'))
                    ]
                
                markdown_text = "Images generated based on the above prompts" if gallery_value else "No images generated yet"
                
                return {
                    preview_image: job_info['image_path'],
                    completion_time: job_info['completion_time'],
                    caption: job_info['caption'],
                    model_name: job_info['model_name'],
                    generated_images: gallery_value
                }
            return None
            
        job_selector.change(
            fn=update_display,
            inputs=[job_selector],
            outputs=[preview_image, completion_time, caption, model_name, generated_images]
        )

        # 修改函数定义，移除 job_id 的关键字参数
        def get_prompts_and_generate(*args):
            pass
            
        run_button.click(
            fn=get_prompts_and_generate,
            inputs=[job_selector, model_name, caption, preview_image],  # 将所有文本框和job_selector作为输入
            outputs=[]
        )

        def select_images(selected_images_list):
            return {selected_images: selected_images_list}

        select_button.click(
            fn=select_images,
            inputs=[generated_images],
            outputs=[selected_images]
        )

    return demo
