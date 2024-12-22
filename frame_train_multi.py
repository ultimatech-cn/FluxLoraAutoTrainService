import gradio as gr
import pandas as pd
import os
from multiprocessing import JoinableQueue
from PIL import Image
from fewshot.fewshot_train import Trainer
from pathlib import Path

def get_done_jobs():
    job_df = pd.read_csv('job_status.csv')
    done_jobs = job_df['jobid'][job_df['status'] == 'done'][job_df['job_type'] == 'ONESHOT_GEN'].tolist()
    return done_jobs

selected_images = []
image_paths = []

def toggle_selection(evt: gr.SelectData):
    global selected_images
    if evt.index in selected_images:
        gr.Warning("Image already selected")
    else:
        selected_images.append(evt.index)
    # 先对selected_images排序
    selected_images.sort()
    print(selected_images)
    return [[image_paths[i] for i in selected_images], selected_images]

def remove_selected(evt: gr.SelectData):
    global selected_images
    img_path = evt.value['image']['path'].split('\\')[-1]
    print(img_path)
    index = -1  # 默认值为 -1 表示未找到
    print(image_paths)
    for i, path in enumerate(image_paths):
        if img_path in path:
            index = i
            break
    print(index)
    if index in selected_images:
        selected_images.remove(index)
    selected_images.sort()
    print(selected_images)
    result_images = [image_paths[i] for i in selected_images]
    return [result_images, selected_images]

# 添加新的全选函数
def select_all():
    global selected_images
    selected_images = list(range(len(image_paths)))  # 选择所有图片的索引
    return [[image_paths[i] for i in selected_images], selected_images]

# 添加新的清除函数
def clear_selection():
    global selected_images
    selected_images = []
    return [[], []]

# Main interface building function
def train_multi_input(queue, job_status_manager):
    trainer = Trainer(queue, job_status_manager)

    state = gr.State(value=selected_images)  # 定义状态变量

    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column(scale=1, min_width=300):
                # done_jobs = get_done_jobs()
                job_selector = gr.Dropdown(
                    choices=["Initial 1"],
                    label="Select JobID of completed model from step 1",
                    multiselect=False,
                    value=None,
                    allow_custom_value=True
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
                show_download_button=False,
                interactive=False,
                allow_preview=False
            )
        with gr.Row():
            all_selected_button = gr.Button('All Selected')
            clear_button = gr.Button('Clear')

        with gr.Accordion("Selected Images", open=False) as selected_sample:
            selected_images_gallery = gr.Gallery(
                label=None,
                show_label=False,
                columns=5,
                object_fit="contain",
                show_download_button=False,
                interactive=False,
                allow_preview=False
            )

        with gr.Row():
            run_button = gr.Button('Start Training', variant="p")


        generated_images.select(fn=toggle_selection, inputs=None, outputs=[selected_images_gallery, state])
        selected_images_gallery.select(fn=remove_selected, inputs=None, outputs=[selected_images_gallery, state])

        # 修改按钮事件处理
        all_selected_button.click(
            fn=select_all,
            inputs=[],
            outputs=[selected_images_gallery, state]
        )

        clear_button.click(
            fn=clear_selection,
            inputs=[],
            outputs=[selected_images_gallery, state]
        )

        def update_display(selected_job):
            global image_paths  # 声明 image_paths 为全局变量
            if selected_job:
                job_df = pd.read_csv("job_status.csv")
                job_info = job_df[((job_df['jobid'] == selected_job) & (job_df['job_type'] == 'ONESHOT_GEN'))].iloc[0]
                generated_dir = f'job_data/{selected_job}/oneshot_generate'
                # get the absolute path of generated_dir
                generated_dir = os.path.abspath(generated_dir)
                gallery_value = []
                if os.path.exists(generated_dir):
                    gallery_value = [
                        os.path.join(generated_dir, img)
                        for img in os.listdir(generated_dir)
                        if img.lower().endswith(('.png', '.jpg', '.jpeg'))
                    ]

                # print(gallery_value)
                image_paths = gallery_value  # 现在这行代码会更新全局变量 image_paths

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

        def load_dropdown_content():
            # 这里可以定义加载下拉菜单内容的逻辑
            done_jobs = get_done_jobs()
            return done_jobs

        # job_selector.focus(
        #     fn=load_dropdown_content,
        #     inputs=[],
        #     outputs=[job_selector]
        # )

        run_button.click(
            fn=trainer.run,
            inputs=[job_selector, selected_images_gallery, model_name, caption, preview_image, state],
            outputs=[]
        )

        def update_dropdown():
            new_choices = get_done_jobs()
            print(new_choices)
            # 使用 gr.Dropdown.update() 方法更新下拉框的选项
            return gr.Dropdown(choices=new_choices, value=new_choices[0])

        demo.load(fn=update_dropdown, inputs=[], outputs=job_selector)

    return demo
