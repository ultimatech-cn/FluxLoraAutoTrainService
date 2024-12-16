from oneshot.BaseTrainer import BaseTrainer
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
logger = setup_logger('AIToolkitFewShotTrainer')

# Add ai-toolkit project path to environment variables
TOOLKIT_PATH = Path(__file__).parent.parent.parent.parent.joinpath("ai-toolkit")
sys.path.insert(0, str(TOOLKIT_PATH))

# Now we can import modules from ai-toolkit normally
from toolkit.job import get_job

# Load project config
from FluxLoraAutoTrainService.common_tools import project_config

if project_config['flux_model_type'] == "FLUX.1-schnell":
    BASE_YAML_PATH = Path(__file__).parent.parent.joinpath('assets').joinpath('ai_toolkit_base_config_schnell.yaml')
else:
    BASE_YAML_PATH = Path(__file__).parent.parent.joinpath('assets').joinpath('ai_toolkit_base_config_dev.yaml') # dev model

class AIToolkitFewShotTrainer(BaseTrainer):

    def __init__(self):
        super().__init__(user_config=None)
        self.yaml_config = self.load_config(BASE_YAML_PATH)

    def load_config(self, config_path: str):
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
        
    def fix_yaml_config(self, yaml_config: dict, user_config: dict):
        # Update the config with user inputs
        yaml_config["config"]["name"] = user_config['model_name']
        yaml_config["config"]["process"][0]["datasets"][0]["folder_path"] = user_config['job_folder'] + os.sep + 'fewshot_dataset'
        yaml_config["config"]["process"][0]["training_folder"] = user_config['job_folder']+ os.sep + 'fewshot_output'

        yaml_config["meta"]["name"] = user_config['model_name']
        return yaml_config

    def create_dataset(self, config: dict):
        print("Creating dataset")

        selected_indexes = config['selected_indexes'] # 0, 1, 2, 3, 4
        selected_images = config['selected_images'] # ['image_path_1', 'image_path_2', 'image_path_3', 'image_path_4', 'image_path_5']
        
        if len(selected_indexes) != len(selected_images):
            logger.error("selected_indexes and selected_images length mismatch")
            return None

        job_folder = str(f"job_data/{config['job_id']}")
        if not os.path.exists(job_folder):
            os.makedirs(job_folder)

        destination_folder = str(f"{job_folder}/fewshot_dataset")
        prompts_file = str(f"{job_folder}/prompts.txt")
        if not os.path.exists(destination_folder):
            raise Exception("fewshot_dataset folder not found")
        if not os.path.exists(prompts_file):
            raise Exception("prompts.txt not found")

        jsonl_file_path = os.path.join(destination_folder, "metadata.jsonl")
        with open(jsonl_file_path, "a") as jsonl_file:
            with open(prompts_file, "r") as prompts_file:
                prompts = prompts_file.readlines()
                for index, image in zip(selected_indexes, selected_images):
                    new_image_path = shutil.copy(image, destination_folder)
                    file_name = os.path.basename(new_image_path)
                    data = {"file_name": file_name, "prompt": prompts[index].strip()}
                    jsonl_file.write(json.dumps(data) + "\n")

        return job_folder

    def train(self, task_data: dict) -> bool:
        self.config = task_data
        # Update task status to Processing
        self.job_manager.update_job_status(self.config['job_id'], 'FEWSHOT_TRAIN', JobStatus.Processing.value)
        # 1. create dataset
        self.config['job_folder'] = self.create_dataset(self.config)
        # 2. get customized yaml config and save it to job data folder
        self.fixed_yaml_config = self.fix_yaml_config(self.yaml_config, self.config) # fix yaml config
        config_path = f"{self.config['job_folder']}/{self.config['job_id']}-{self.config['model_name']}-fewshot.yaml"
        with open(config_path, "w") as f:
            yaml.dump(self.fixed_yaml_config, f)
            
        # run the job locally
        job = get_job(config_path)
        try:
            job.run()
            job.cleanup()
            # time.sleep(100) # mock training time
            self.job_manager.update_job_status(self.config['job_id'], 'FEWSHOT_TRAIN', JobStatus.Done.value)
            return True
        except Exception as e:
            job.cleanup()
            logger.error(f"Training failed: {str(e)}")
            self.job_manager.update_job_status(self.config['job_id'], 'FEWSHOT_TRAIN', JobStatus.Failed.value)

    def change_job_status(self, job_id: str, job_type: str, status: str):
        self.job_manager.update_job_status(job_id, job_type, status)