import time
from logger_config import setup_logger
from job_status import JobStatus
from oneshot_trainer.AIToolkitTrainer import AIToolkitTrainer
import gradio as gr

logger = setup_logger('consumer')

def consumer(q):
    aitoolkit_trainer = AIToolkitTrainer()
    while True:
        try:
            # Get data from queue
            task = q.get()

            print(f"Consumer received data: {task}")
            if task:
                try:
                    logger.info(f"Task processing: {task['job_id']}")
                    if aitoolkit_trainer.train(task_data=task):
                        gr.Info(f"Task completed successfully: {task['job_id']}")
                        logger.info(f"Task completed successfully: {task['job_id']}")
                except Exception as e:
                    logger.error(f"Error processing task {task['job_id']}: {str(e)}")
                finally:
                    # 标记任务完成，这样队列知道可以处理下一个任务
                    q.task_done()
            
            time.sleep(10)
        except Exception as e:
            logger.error(f"Consumer error: {str(e)}")
            aitoolkit_trainer.change_job_status(task['job_id'], JobStatus.Failed.value)
            time.sleep(10)