import gradio as gr
from pathlib import Path
import pandas as pd
from oneshot.oneshot_gen import Generator
import os

def get_done_jobs():
    job_df = pd.read_csv('job_status.csv')
    done_jobs = job_df['jobid'][job_df['status'] == 'done'][job_df['job_type'] == 'ONESHOT_TRAIN'].tolist()
    return done_jobs

def load_prompts(gender) -> list[str]:
    file_path = f"assets/image_prompts_for_{gender}.txt"
    with open(file_path, 'r') as file:
        prompts = file.readlines()
    return [prompt.strip() for prompt in prompts]

prompts_mans = load_prompts('man')

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
                gender = gr.Radio(
                    choices=["man", "lady"],
                    label="性别选择",
                    value="man"
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
            for i in range(len(prompts_mans)):
                # prefix = caption.value.split(' ')[0] if caption.value else None
                grs.append(gr.Textbox(label="prompt " + str(i + 1), value=None))

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
                    prompts_mans[i].replace('[trigger]', prefix) if prefix else prompts_mans[i]
                    for i in range(len(prompts_mans))
                ]
                
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

        def update_prompts_based_on_gender(selected_gender):
            prompts = load_prompts(selected_gender)
            return {gr_prompt: prompt for gr_prompt, prompt in zip(grs, prompts)}

        gender.change(
            fn=update_prompts_based_on_gender,
            inputs=[gender],
            outputs=grs
        )

    return demo
