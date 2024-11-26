## FluxLoraAutoTrainService

### Environment and Running Instructions:

#### 1. Project Location
Please place this project under the ai-toolkit directory

#### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 3. Run Service
```bash
python main.py
```

There are two config files on subfolder "assets":
1. ai_toolkit_base_config.yaml is template file for lora training setting.
2. project_config.yaml is setting for queue size only so far.
