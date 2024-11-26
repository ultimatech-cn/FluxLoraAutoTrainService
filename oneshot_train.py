from logger_config import setup_logger
import gradio as gr
import os
from uuid import uuid4
from job_status import JobStatus
from multiprocessing import JoinableQueue
from job_record_tools import JobStatusManager

# Get logger instance
logger = setup_logger('oneshot_train')

# Set directory for saving images
UPLOAD_DIR = "job_data"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

class Trainer:
    def __init__(self, queue: JoinableQueue, job_status_manager: JobStatusManager):
        self.image_path = None  # Path to training image
        self.job_id = None      # Task identifier
        self.model_name = None  # Name of the model
        self.caption = None      # Training prompt
        self.queue = queue      # Queue for tasks
        self.job_manager = job_status_manager   # Job status manager
    def run(
            self,
            model_name: str,
            instance_images: list,
            trigger_words: str,
            caption: str
    ) -> str:
        # Validate instance
        if instance_images is None or len(instance_images) == 0:
            gr.Warning('Please upload photos!')
            return [[], self.job_manager.get_all_records()]

        # Check image count
        if len(instance_images) > 1:
            gr.Warning('Only one photo can be trained!')
            return [[], self.job_manager.get_all_records()]
        
        # Generate unique job ID
        self.job_id = str(uuid4().hex)

        # Store user input information
        self.model_name = model_name
        self.image_path = instance_images[0][0]
        self.caption = caption.replace("[trigger]", trigger_words)
        
        # Prepare task data for queue
        task_data = {
                "job_id": self.job_id,
                "model_name": model_name,
                "image_path": self.image_path,
                "caption": self.caption
        }
        
        # Use put_nowait with exception handling to check if queue is full
        try:
            if not self.queue.full():
                self.job_manager.add_job(self.job_id, self.image_path, self.caption, self.model_name, JobStatus.WaitingQueue.value)
                self.queue.put_nowait(task_data)


                gr.Info("Job has been added to queue, please wait patiently")
                logger.info(f"Task {task_data['job_id']} added to queue successfully")
                return [[], self.job_manager.get_all_records()]
            else:
                logger.info(f"Task {task_data['job_id']} failed to add to queue")
                gr.Warning("Queue is full. Maximum of 5 tasks allowed. Please try again later.")
                return [instance_images, self.job_manager.get_all_records()]
        except Exception as e:
            gr.Error(f"An error occurred: {e}")