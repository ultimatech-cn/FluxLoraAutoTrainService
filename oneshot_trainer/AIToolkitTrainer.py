from oneshot_trainer.BaseTrainer import BaseTrainer
import yaml
import os
import sys
from pathlib import Path
from job_status import JobStatus
import shutil
import json
import time
# Get logger instance
from logger_config import setup_logger
logger = setup_logger('AIToolkitTrainer')

# Add ai-toolkit project path to environment variables
TOOLKIT_PATH = Path(__file__).parent.parent.parent.parent.joinpath("ai-toolkit")
sys.path.insert(0, str(TOOLKIT_PATH))

# Now we can import modules from ai-toolkit normally
#from toolkit.job import get_job

BASE_YAML_PATH = Path(__file__).parent.parent.joinpath('assets').joinpath('ai_toolkit_base_config.yaml')


class AIToolkitTrainer(BaseTrainer):

    def __init__(self):
        super().__init__(user_config=None)
        self.yaml_config = self.load_config(BASE_YAML_PATH)

    def load_config(self, config_path: str):
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)

    def fix_yaml_config(self, yaml_config: dict, user_config: dict):
        # Update the config with user inputs
        yaml_config["config"]["name"] = user_config['model_name']
        yaml_config["config"]["process"][0]["datasets"][0]["folder_path"] = user_config['job_folder'] + os.sep + 'oneshot_dataset'
        yaml_config["config"]["process"][0]["training_folder"] = user_config['job_folder']+ os.sep + 'output'

        yaml_config["meta"]["name"] = user_config['model_name']
        return yaml_config

    def create_dataset(self, config: dict):
        print("Creating dataset")
        # os.makedirs(f"job_data/{config['job_id']}", exist_ok=True)

        image = config['image_path']
        caption = config['caption']

        job_folder = str(f"job_data/{config['job_id']}")
        if not os.path.exists(job_folder):
            os.makedirs(job_folder)

        destination_folder = str(f"{job_folder}/oneshot_dataset")
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)

        jsonl_file_path = os.path.join(destination_folder, "metadata.jsonl")
        with open(jsonl_file_path, "a") as jsonl_file:
            new_image_path = shutil.copy(image, destination_folder)
            file_name = os.path.basename(new_image_path)
            data = {"file_name": file_name, "prompt": caption}
            jsonl_file.write(json.dumps(data) + "\n")

        return job_folder

    def train(self, task_data: dict) -> bool:
        self.config = task_data
        # Update task status to Processing
        self.job_manager.update_job_status(self.config['job_id'], JobStatus.Processing.value)
        # 1. create dataset
        self.config['job_folder'] = self.create_dataset(self.config)
        # 2. get customized yaml config and save it to job data folder
        self.fixed_yaml_config = self.fix_yaml_config(self.yaml_config, self.config) # fix yaml config
        config_path = f"{self.config['job_folder']}/{self.config['job_id']}-{self.config['model_name']}.yaml"
        with open(config_path, "w") as f:
            yaml.dump(self.fixed_yaml_config, f)

        # run the job locally
        job = get_job(config_path)
        try:
            job.run()
            job.cleanup()
            # time.sleep(100) # mock training time
            self.job_manager.update_job_status(self.config['job_id'], JobStatus.Done.value)
            return True
        except Exception as e:
            job.cleanup()
            logger.error(f"Training failed: {str(e)}")
            self.job_manager.update_job_status(self.config['job_id'], JobStatus.Failed.value)

    def change_job_status(self, job_id: str, status: str):
        self.job_manager.update_job_status(job_id, status)
