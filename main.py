import gradio as gr
import os
from oneshot.oneshot_train import Trainer
import pandas as pd
import base64
from multiprocessing import Process, JoinableQueue
from consumer import consumer
from frame_train import train_input
from frame_train_multi import train_multi_input
from frame_inference import train_inference
from job_record_tools import JobStatusManager

'''
Load project config
'''
from common_tools import project_config

queue = JoinableQueue(maxsize=project_config['queue_size'])
job_status_manager = JobStatusManager()

def init_job():
    '''
    Initialize job
    Check the first queue_size records in job_status.csv, if status is WaitingQueue or Processing,
    add these tasks to the queue
    '''
    # v1.1 新增的CSV检查和更新逻辑, 为了适配上次1.0版本的数据
    csv_path = 'job_status.csv'
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        if len(df) > 0 and 'job_type' not in df.columns:
            # 在第三列位置插入job_type列
            df.insert(2, 'job_type', 'ONESHOT_TRAIN')
            df.to_csv(csv_path, index=False)
    
    print(f"init job")
    records = job_status_manager.get_pending_jobs()
    print(f"update queue: {records}")
    if records is not None and len(records) > 0: 
        for record in records:
            if record['job_type'] == 'ONESHOT_TRAIN':
                queue.put_nowait(record)
            elif record['job_type'] == 'ONESHOT_GEN':
                # 读取job_data/jobid目录下的prompts.txt文件内容
                prompts_path = os.path.join('job_data', record['job_id'], 'prompts.txt')
                if os.path.exists(prompts_path):
                    with open(prompts_path, 'r') as f:
                        prompts = f.read().splitlines()
                    record['prompts'] = prompts
                    print(f"record: {record}")
                queue.put_nowait(record)


# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# <center> \N{fire} One Image Flux Training</center>")
    with gr.Tabs():
        with gr.TabItem('\N{rocket}Step 1 (One Image Train)'):
            train_input(queue, job_status_manager)
        with gr.TabItem('\N{party popper}Step2 (Generating Human model Images)'):
            train_inference(queue, job_status_manager)
        with gr.TabItem('\N{rocket}Step3 (Multiple Image Train)'):
            train_multi_input(queue, job_status_manager)

if __name__ == "__main__":
    # Start consumer process first
    consumer_process = Process(target=consumer, args=(queue,))
    consumer_process.start()

    init_job()

    # Then launch Gradio interface
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)