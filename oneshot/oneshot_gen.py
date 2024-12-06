from logger_config import setup_logger
import gradio as gr
import os
from uuid import uuid4
from job_status import JobStatus
from multiprocessing import JoinableQueue
from job_record_tools import JobStatusManager

# Get logger instance
logger = setup_logger('oneshot_gen')

# Set directory for saving images
UPLOAD_DIR = "job_data"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

class Generator:
    def __init__(self, queue: JoinableQueue, job_status_manager: JobStatusManager):
        self.queue = queue      # Queue for tasks
        self.job_manager = job_status_manager   # Job status manager
    def run(
            self,
            grs: list[str], 
            job_id: str,
            model_name: str,
            caption: str,
            preview_image
    ) -> str:
        # Validate 
        if job_id is None or len(job_id) == 0:
            gr.Warning('Please select a jobid to generate the task!')
            return None
        image_path = preview_image

        # Prepare task data for queue
        task_data = {
                "job_id": job_id,
                "job_type": "ONESHOT_GEN",
                "model_name": model_name,
                "image_path": image_path,
                "caption": caption,
                "prompts": grs
        }
        
        # Use put_nowait with exception handling to check if queue is full
        try:
            if not self.queue.full():
                self.job_manager.add_job(job_id, image_path, "ONESHOT_GEN", caption, model_name, JobStatus.WaitingQueue.value)
                self.queue.put_nowait(task_data)
                gr.Info("Task has been added to queue, please wait patiently")
                logger.info(f"Task {task_data['job_id']} oneshot gen task added to queue successfully")
            else:
                logger.info(f"Task {task_data['job_id']} oneshot gen task failed to add to queue")
                gr.Warning("Queue is full. Maximum of 5 tasks allowed. Please try again later.")
            return None
        except Exception as e:
            gr.Error(f"An error occurred: {e}")