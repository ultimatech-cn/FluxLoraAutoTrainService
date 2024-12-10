from oneshot.BaseTrainer import BaseTrainer
import yaml
import sys
from pathlib import Path
from job_status import JobStatus
from job_record_tools import JobStatusManager
import os
import shutil
# Get logger instance
from logger_config import setup_logger
logger = setup_logger('AIToolkitOneShotGen')

# Add ai-toolkit project path to environment variables
TOOLKIT_PATH = Path(__file__).parent.parent.parent.parent.joinpath("ai-toolkit")
sys.path.insert(0, str(TOOLKIT_PATH))

# Now we can import modules from ai-toolkit normally
from toolkit.job import get_job

# Load project config
from FluxLoraAutoTrainService.common_tools import project_config

if project_config['flux_model_type'] == "FLUX.1-schnell":
    BASE_GEN_YAML_PATH = Path(__file__).parent.parent.joinpath('assets').joinpath('ai_toolkit_base_generatic_config_schnell.yaml')
else:
    BASE_GEN_YAML_PATH = Path(__file__).parent.parent.joinpath('assets').joinpath('ai_toolkit_base_generatic_config_dev.yaml') # dev model

BASE_OUTPUT_PATH = Path(__file__).parent.parent.joinpath('job_data')

class AIToolkitOneShotGen():

    def __init__(self):
        self.job_manager = JobStatusManager()
        self.yaml_config = self.load_config(BASE_GEN_YAML_PATH)
        self.job_path = None

    def load_config(self, config_path: str):
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
        
    def fix_yaml_config(self, yaml_config: dict, prompts: list[str], model_name: str, output_path: str):
        # Update the config with user inputs
        yaml_config["config"]["process"][0]['output_folder'] = output_path
        yaml_config["config"]["process"][0]['model']['lora_path'] = str(self.job_path.joinpath('output').joinpath(model_name).joinpath(model_name + '.safetensors').resolve())
        # 直接替换整个 prompts 列表，而不是逐个赋值
        yaml_config["config"]["process"][0]['generate']['prompts'] = prompts
        return yaml_config

    def gen(self, config_data: dict) -> str:

        # Update task status to Processing
        self.job_manager.update_job_status(config_data['job_id'], 'ONESHOT_GEN', JobStatus.Processing.value)
        self.job_path = BASE_OUTPUT_PATH.joinpath(config_data['job_id'])
        output_path = str(self.job_path.joinpath('oneshot_generate').resolve())
        if os.path.exists(output_path):
            shutil.rmtree(output_path) # if exists, delete the folder
        self.fixed_yaml_config = self.fix_yaml_config(self.yaml_config, config_data['prompts'], config_data['model_name'], output_path) # fix yaml config
        config_path =  str(self.job_path.joinpath(f"{config_data['job_id']}-{config_data['model_name']}-gen.yaml").resolve())
        print(config_path)

        # save the fixed yaml config to the job path
        with open(config_path, "w") as f:
            yaml.dump(self.fixed_yaml_config, f)

        # save prompts to the job path
        with open(self.job_path.joinpath('prompts.txt'), "w") as f:
            for prompt in config_data['prompts']:
                f.write(prompt + "\n")
            
        # run the job locally
        job = get_job(config_path)
        try:
            job.run()
            job.cleanup()
            self.job_manager.update_job_status(config_data['job_id'], 'ONESHOT_GEN', JobStatus.Done.value)
            return True
        except Exception as e:
            job.cleanup()
            logger.error(f"Generation failed: {str(e)}")
            self.job_manager.update_job_status(config_data['job_id'], 'ONESHOT_GEN', JobStatus.Failed.value)
        
        return False

    def change_job_status(self, job_id: str, job_type: str, status: str):
        self.job_manager.update_job_status(job_id, job_type, status)