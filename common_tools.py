import yaml

def load_config(file_path="config.yaml"):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
        print(f"Configuration loaded successfully: {config}")
        return config
    except Exception as e:
        print(f"Failed to load configuration: {e}")
        return {"error": str(e)}

project_config = load_config('./assets/project_config.yaml')
