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


## Usage
Step 1：
 upload 1 images for character, the image must be high resolution. The face should be clear and detailed.

Step 2:
  choose a task from "step 1" and gender of the character.
  There two files containing prompts for character generation, which are ".\assets\image_prompts_for_lady.txt" and ".\assets\image_prompts_for_man.txt"
  You can input you own prompt in the UI or change the above txt files.
  Click "Start Generation"
Step 3：  
  Choose a task from "step 2" and select generated images to selected images. You can click "start Training"

### Configration files
The files are under the folder ".\assets"
1. project_config.yaml is for application. You can set queue size and default training flux model(dev or schenell).
2. ai_toolkit_base_config_dev.yaml  is used at "step 1" for one-shot training
3. image_prompts_for_lady.txt and image_prompts_for_man.txt are used in "step 2", there are pre-defined prompts list. you can add and updating prompt in these two files.
4. ai_toolkit_base_generatic_config_dev is used for training in "step 3"
