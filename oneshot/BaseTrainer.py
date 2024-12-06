from job_record_tools import JobStatusManager

class BaseTrainer(object):

    def __init__(self, user_config: dict):
        self.job_manager = JobStatusManager()
        self.user_config = user_config # config includes job_id, model_name, image_path, caption, dataset_folder, etc.

    def train(self) -> bool:
        pass

    def get_conf(self, key, default=None, required=False):
        if key in self.config:
            return self.config[key]
        elif required:
            raise ValueError(f'config file error. Missing "config.{key}" key')
        else:
            return default