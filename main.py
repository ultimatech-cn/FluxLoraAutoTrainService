import gradio as gr
import os
from oneshot_train import Trainer
import pandas as pd
import base64
from multiprocessing import Process, JoinableQueue
from consumer import consumer
from frame_train import train_input
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
    Check the first 10 records in job_status.csv, if status is WaitingQueue or Processing,
    add these tasks to the queue
    '''
    print(f"init job")
    records = job_status_manager.get_pending_jobs()
    print(f"update queue: {records}")
    for record in records:
        queue.put_nowait(record)


# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# <center> \N{fire} One Image Flux Training</center>")
    with gr.Tabs():
        with gr.TabItem('\N{rocket}Step 1 (One Image Train)'):
            train_input(queue, job_status_manager)
            pass
        with gr.TabItem('\N{party popper}Step2 (Generating Human model Images)'):
            # inference_input()
            pass

if __name__ == "__main__":
    # Start consumer process first
    consumer_process = Process(target=consumer, args=(queue,))
    consumer_process.start()

    init_job()

    # Then launch Gradio interface
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)