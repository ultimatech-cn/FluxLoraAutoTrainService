import gradio as gr

# 定义图片路径和标签
image_paths = ["image1.jpg", "image2.jpg", "image3.jpg"]
image_labels = ["图片 1", "图片 2", "图片 3"]

def on_images_selected(selected_labels):
    if selected_labels:
        return f"您选择了以下图片：{', '.join(selected_labels)}"
    else:
        return "您未选择任何图片。"

with gr.Blocks() as demo:
    with gr.Row():
        gallery = gr.Gallery(
            value=[(path, label) for path, label in zip(image_paths, image_labels)],
            label="图片画廊",
            show_label=True,
            columns=3
        )
    checkbox_group = gr.CheckboxGroup(
        choices=image_labels,
        label="选择图片"
    )
    result = gr.Textbox(label="选择结果")

    checkbox_group.change(on_images_selected, inputs=checkbox_group, outputs=result)

demo.launch()
