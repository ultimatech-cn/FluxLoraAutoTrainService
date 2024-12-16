from logger_config import setup_logger
import gradio as gr
import os
from uuid import uuid4
from job_status import JobStatus
from multiprocessing import JoinableQueue
from job_record_tools import JobStatusManager
from pathlib import Path
import shutil
# Get logger instance
logger = setup_logger('fewshot_train')

# Set directory for saving images
UPLOAD_DIR = Path(__file__).parent.parent.joinpath('job_data')
print(f"UPLOAD_DIR: {UPLOAD_DIR}")
if not os.path.exists(str(UPLOAD_DIR)):
    os.makedirs(str(UPLOAD_DIR))

class Trainer:
    def __init__(self, queue: JoinableQueue, job_status_manager: JobStatusManager):
        self.image_path = None  # Path to training image
        self.job_id = None      # Task identifier
        self.model_name = None  # Name of the model
        self.caption = None      # Training prompt
        self.queue = queue      # Queue for tasks
        self.job_manager = job_status_manager   # Job status manager
        self.selected_images = []
    def run(
            self,
            job_id: str,
            selected_images: list,
            model_name: str,
            caption: str,
            preview_image: str,
            selected_indexes: list
    ) -> str:
        # Validate instance
        if selected_images is None or len(selected_images) == 0:
            gr.Warning('Please select images!')
            return None

        # Check image count
        if len(selected_images) < 3:
            gr.Warning('At least 2 images are required!')
            return None
        
        self.job_id = job_id
        self.model_name = model_name
        self.image_path = preview_image
        self.caption = caption
        self.selected_images = [item[0] for item in selected_images]
        self.selected_indexes = selected_indexes

        # check job status
        if self.job_manager.check_job_status(self.job_id, "FEWSHOT_TRAIN", JobStatus.Processing.value) or self.job_manager.check_job_status(self.job_id, "FEWSHOT_TRAIN", JobStatus.WaitingQueue.value):
            gr.Warning("Task is already in progress. Please wait for it to complete.")
            return None

        # Prepare task data for queue
        task_data = {
                "job_id": self.job_id,
                "job_type": "FEWSHOT_TRAIN",
                "model_name": self.model_name,
                "image_path": self.image_path,
                "caption": self.caption,
                "selected_images": self.selected_images,
                "selected_indexes": self.selected_indexes
        }
        
        # Use put_nowait with exception handling to check if queue is full
        try:
            if not self.queue.full():
                self.job_manager.add_job(self.job_id, self.image_path, 'FEWSHOT_TRAIN', self.caption, self.model_name, JobStatus.WaitingQueue.value)
                self.queue.put_nowait(task_data)

                gr.Info("Job has been added to queue, please wait patiently")
                logger.info(f"Task {task_data['job_id']} added to queue successfully")
                return None
            else:
                logger.info(f"Task {task_data['job_id']} failed to add to queue")
                gr.Warning("Queue is full. Maximum of 5 tasks allowed. Please try again later.")
                return None
        except Exception as e:
            gr.Error(f"An error occurred: {e}")