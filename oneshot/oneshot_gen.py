from logger_config import setup_logger
import gradio as gr
import os
from uuid import uuid4
from job_status import JobStatus
from multiprocessing import JoinableQueue
from job_record_tools import JobStatusManager
from pathlib import Path
# Get logger instance
logger = setup_logger('oneshot_gen')

# Set directory for saving images
UPLOAD_DIR = Path(__file__).parent.parent.joinpath('job_data')
print(f"UPLOAD_DIR: {UPLOAD_DIR}")
if not os.path.exists(str(UPLOAD_DIR)):
    os.makedirs(str(UPLOAD_DIR))

def is_different_prompts(grs: list[str], job_path: str) -> bool:
    # 读取job_path下的prompts.txt
    with open(Path(job_path).joinpath('prompts.txt'), 'r') as f:
        saved_prompts = f.read().splitlines()
    return grs != saved_prompts

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

        # validate the job_id
        if job_id is None or len(job_id) == 0:
            gr.Warning('Please select a jobid to generate the task!')
            return None
        
        # check job status
        if self.job_manager.check_job_status(job_id, "ONESHOT_GEN", JobStatus.Processing.value) or self.job_manager.check_job_status(job_id, "ONESHOT_GEN", JobStatus.WaitingQueue.value):
            gr.Warning("Task is already in progress. Please wait for it to complete.")
            return None

        if self.job_manager.check_job_status(job_id, "ONESHOT_GEN", JobStatus.Done.value):
            # 弹出一个对话框，询问用户是否要继续提交生成任务
            if not gr.Info("该任务已生成完成，是否需要重新生成?").then(lambda: True):
                gr.Warning("已取消重新生成任务")
                return None
            # 如果prompts相同，提示用户prompts和上次的的一致，不需要重新生成
            if is_different_prompts(grs, str(UPLOAD_DIR.joinpath(job_id))):
                gr.Warning("Prompts are the same as the last generation, no need to regenerate.")
                return None
            
        # 将prompts写入job_data/job_id/prompts.txt
        with open(Path(UPLOAD_DIR).joinpath(job_id).joinpath('prompts.txt'), 'w') as f:
            f.write('\n'.join(grs))

        # Prepare task data for queue
        task_data = {
                "job_id": job_id,
                "job_type": "ONESHOT_GEN",
                "model_name": model_name,
                "image_path": preview_image,
                "caption": caption,
                "prompts": grs
        }
        
        # Use put_nowait with exception handling to check if queue is full
        try:
            if not self.queue.full():
                self.job_manager.add_job(job_id, preview_image, "ONESHOT_GEN", caption, model_name, JobStatus.WaitingQueue.value)
                self.queue.put_nowait(task_data)
                gr.Info("Task has been added to queue, please wait patiently")
                logger.info(f"Task {task_data['job_id']} oneshot gen task added to queue successfully")
            else:
                logger.info(f"Task {task_data['job_id']} oneshot gen task failed to add to queue")
                gr.Warning("Queue is full. Maximum of 5 tasks allowed. Please try again later.")
            return None
        except Exception as e:
            gr.Error(f"An error occurred: {e}")