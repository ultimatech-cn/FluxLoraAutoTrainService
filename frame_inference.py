import gradio as gr
from pathlib import Path
import pandas as pd
from oneshot.oneshot_gen import Generator
import os

init_prompt = [
                "[trigger] in a garden",
                "[trigger] in a coffee shop",
                "[trigger] in a library",
                "[trigger] at the beach",
                "[trigger] in a park",
                "[trigger] in an office",
                "[trigger] at a party",
                "[trigger] in a restaurant",
                "[trigger] at home",
                "[trigger] in a gym",
                "[trigger] on a mountain",
                "[trigger] in a museum",
                "[trigger] at a concert",
                "[trigger] in a shopping mall",
                "[trigger] on a street",
                "[trigger] in a classroom",
                "[trigger] at a sports event",
                "[trigger] in a studio",
                "[trigger] at an airport",
                "[trigger] in a forest"
            ]

def get_done_jobs():
    job_df = pd.read_csv('job_status.csv')
    done_jobs = job_df['jobid'][job_df['status'] == 'done'][job_df['job_type'] == 'ONESHOT_TRAIN'].tolist()
    return done_jobs

# Main interface building function
def train_inference(queue, job_status_manager):

    img_gen = Generator(queue, job_status_manager)
   
    with gr.Blocks(css="frame_train.css") as demo:

        done_jobs = get_done_jobs()
        
        # Create dropdown and display components
        with gr.Row():
            with gr.Column(scale=1, min_width=300):
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
            

        with gr.Accordion("Sample prompts", open=False) as sample:
            gr.Markdown(
                "Include sample prompts to test out your trained model."
            )

            
            grs = []
            for i in range(len(init_prompt)):
                prefix = caption.value.split(' ')[0] if caption.value else None
                prompt_value = f"{init_prompt[i].replace('[trigger]', prefix)}" if prefix else init_prompt[i]
                grs.append(gr.Textbox(label="prompt " + str(i + 1), value=prompt_value))

        with gr.Accordion("Generated Images", open=False) as sample:
            img_markdown = gr.Markdown(
                "Images generated based on the above prompts"
            )
            generated_images = gr.Gallery(
                label=None, 
                show_label=False,
                columns=5,
                object_fit="contain"
            )

        with gr.Row():
            run_button = gr.Button('Start Generation', variant="primary")
            
        def update_display(selected_job):
            if selected_job:
                job_df = pd.read_csv("job_status.csv")
                job_info = job_df[job_df['jobid'] == selected_job].iloc[0]
                
                prefix = job_info['caption'].split(' ')[0] if job_info['caption'] else None
                updated_prompts = [
                    init_prompt[i].replace('[trigger]', prefix) if prefix else init_prompt[i]
                    for i in range(len(init_prompt))
                ]
                
                generated_dir = f'job_data/{selected_job}/oneshot_generate'
                if os.path.exists(generated_dir):
                    image_paths = [os.path.join(generated_dir, img) 
                                 for img in os.listdir(generated_dir) 
                                 if img.endswith(('.png', '.jpg', '.jpeg'))]
                    if image_paths:
                        gallery_value = image_paths
                        markdown_text = "Images generated based on the above prompts"
                    else:
                        gallery_value = None
                        markdown_text = "No images generated yet or generation in progress"
                else:
                    gallery_value = None
                    markdown_text = "No images generated yet or generation in progress"
                
                return {
                    preview_image: job_info['image_path'],
                    completion_time: job_info['completion_time'],
                    caption: job_info['caption'],
                    model_name: job_info['model_name'],
                    generated_images: gallery_value,
                    img_markdown: markdown_text,
                    **{gr_prompt: prompt for gr_prompt, prompt in zip(grs, updated_prompts)}
                }
            return None
            
        job_selector.change(
            fn=update_display,
            inputs=[job_selector],
            outputs=[preview_image, completion_time, caption, model_name, generated_images, img_markdown] + grs
        )

        # 修改函数定义，移除 job_id 的关键字参数
        def get_prompts_and_generate(*args):
            # 最后一个参数是 job_id
            *textbox_values, job_id, model_name, caption, preview_image = args
            # 将文本框的值转换为列表
            prompts = list(textbox_values)
            return img_gen.run(prompts, job_id, model_name, caption, preview_image)
            
        run_button.click(
            fn=get_prompts_and_generate,
            inputs=grs + [job_selector, model_name, caption, preview_image],  # 将所有文本框和job_selector作为输入
            outputs=[]
        )

    return demo
