import time
from logger_config import setup_logger
from job_status import JobStatus
from oneshot.AIToolkitOneShotTrainer import AIToolkitOneShotTrainer
from oneshot.AIToolkitOneShotGen import AIToolkitOneShotGen
import gradio as gr

logger = setup_logger('consumer')

def consumer(q):
    oneshot_trainer = AIToolkitOneShotTrainer()
    oneshot_gen = AIToolkitOneShotGen()
    while True:
        try:
            task = q.get()

            print(f"Consumer received data: {task}")
            if task:
                try:
                    if task['job_type'] == "ONESHOT_TRAIN": 
                        logger.info(f"OneShotTrainer task processing: {task['job_id']}")
                        if oneshot_trainer.train(task_data=task):
                            gr.Info(f"OneShotTrainer task completed successfully: {task['job_id']}")
                            logger.info(f"OneShotTrainer task completed successfully: {task['job_id']}")

                    elif task['job_type'] == "ONESHOT_GEN":
                        logger.info(f"OneShotGen task processing: {task['job_id']}")
                        if oneshot_gen.gen(config_data=task):
                            gr.Info(f"OneShotGen task completed successfully: {task['job_id']}")
                            logger.info(f"OneShotGen task completed successfully: {task['job_id']}")
                except Exception as e:
                    logger.error(f"Error processing {task['job_type']} task {task['job_id']}: {str(e)}")
                finally:
                    # 标记任务完成，这样队列知道可以处理下一个任务
                    q.task_done()
            
            time.sleep(10)
        except Exception as e:
            logger.error(f"Consumer error: {str(e)}")
            if task['job_type'] == "ONESHOT_TRAIN":
                oneshot_trainer.change_job_status(task['job_id'], task['job_type'], JobStatus.Failed.value)
            elif task['job_type'] == "ONESHOT_GEN":
                oneshot_gen.change_job_status(task['job_id'], task['job_type'], JobStatus.Failed.value)
            time.sleep(10)