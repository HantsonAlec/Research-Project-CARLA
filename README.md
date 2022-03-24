# Table of contents
- [Installation guide](#installation-guide)
  - [Requirements](#requirements)
  - [Downloads](#downloads)
  - [Dataset](#dataset)
  - [Extra](#extra)
- [User guide](#user-guide) 
  - [Carla](#carla)
  - [Train](#train)
  - [Simulation](#simulation)
  - [Possible problems](#possible-problems)
  - [Acknowledgements](#acknowledgements)

# Installation guide

## Requirements
### Carla
According to the official CARLA documentation, these are the requirements:
- Windows of Linux OS.
- GPU met minimum 6GB geheugen, 8GB is aangeraden. 20GB vrije ruimte.
- Python3 (python 2.7 is ook mogelijk voor linux).
- 2 vrije TCP poorten, Standaard is dit 2000 & 2001

### AI models
To use the AI models it is highly recommended to use anaconda. This installation guide is based on using anaconda but with a few installation adjustments it can also be used without.
Further Prerequisites:
- Pyhton 3(.7)
- Windows of Linux OS.
- If you don't have Linux OS it is recommended to have access to a cloud GPU such like Azure or AWS.

## Downloads
### Download carla
1) Go to the [downloads](https:/github.com/carla-simulator/carla/blob/master/Docs/download.md) page on the official CARLA github.
2) Choose a CARLA release, in this case CARLA 0.9.13.
3) Download the windows or linux version.
    - Remark. This project uses the windows version of CARLA, further installation will rely on this.
4) Extract the zip file. In this folder you can find the simulator.
5) The client library will be installed along with the requirements/enviroment installation.
### Download envoirement
1) Clone this repository to the PythonAPI folder
    - You can find the PythonAPI folder in the same folder where the simulator is located.
2) Create an anaconda envoirement using the envoirement.yml file.
    - ```conda create --file environment.yml```
### Download extern githubs
My repositories also uses some external repositories. Therefore, it is also recommended to clone them.
1) In the main folder(RESEARCH-PROJECT-CARLA)
    - [Carla-Lane-Detection-Dataset-Generation](https://github.com/Glutamat42/Carla-Lane-Detection-Dataset-Generation)
      - To generate a dataset for lane detection.To generate a dataset for lane detection.
    - [LSTR](https://github.com/liuruijin17/LSTR)
      - For the use of help classes in making predictions & training.
2) In the detr train folder(RESEARCH-PROJECT-CARLA/train/detr-pytorch)
    - [DETR](https://github.com/facebookresearch/detr)
      - For evaluation at DETR training.
3) In de yolo train folder(RESEARCH-PROJECT-CARLA/train/yolo)
    - [YOLO](https://github.com/ultralytics/yolov5)
      - For training YOLO model.
 
## Dataset
### DATASET DOWNLOAD/UPLOAD
To retrain the models yourself, it is important to have the necessary data. The data I used to train the models can be found in a zip file on [github](https://github.com/HantsonAlec/Research-Project-CARLA/tree/main/datasets) or on [kaggle](https://www.kaggle.com/alechantson/carladataset). It is highly recommended to upload this data to [roboflow](https://roboflow.com/). Roboflow makes it easy to add new data, apply augmentation, and download data for use.

## Extra
### Path update
1) Move the training file for azure from RESEARCH-PROJECT-CARLA/train/lstr- azure/train_azure.py to RESEARCH-PROJECT-CARLA/LSTR/
2) Move the LSTR model from RESEARCH-PROJECT-CARLA/models/LSTR_500000.pkl to RESEARCH-PROJECT-CARLA/LSTR/cache/nnet/LSTR/.

# User guide
## Carla
### Simulator info
CARLA uses a server-client system. The server is the CARLA simulator that runs and collects data. The client can use the CARLA API to influence the envoirement using python scripts. It is therefore important that the CARLA simulator is always running when you want to start the simulation.

### Start simulator
Starting the CARLA simulator is very easy:
1) Go to the CARLA folder you downloaded according to the installation instructions.
2) Start CARLAUE4.exe

### Change map
The carla simulator by default has many different maps. To change the map, the simulator must already be started (see step 1.1). The map can be changed in the following way:
1) Open the file "CARLA_0.9.13/PythonAPI/Research-Project-CARLA/layers.py"
2) On line 17, you can customize the directory. All possibilities can be found in the [CARLA documentation](https://carla.readthedocs.io/en/latest/core_map/#carla-maps).
    - It is currently set to Town02_Opt by default.
3) Run layers.py and you will see that the folder changes.

### Simulate traffic
To make the simulation as realistic as possible it is possible to simulate traffic. This can be done using a script that carla provides herself.
1) Open a terminal and go to "CARLA_0.9.13/PythonAPI/examples"
2) Run generate_traffic.py and the traffic will start.
    - Number of vehicles and pedestrians can be adjusted if needed with parameters -n for "number of vechicles" and -w for "number of walkers"

## Data collection
The data I used is publicly available (see installation manual). If you prefer to collect data yourself you can do so in the following way.
### Object detection data
1) Open a terminal and go to “CARLA_0.9.13/PythonAPI/Research-Project-CARLA”
2) Run collect_data.py
3) The image data will now be found in the output folder. (“CARLA_0.9.13/PythonAPI/Research- Project-CARLA/output”)
4) The data annotation can be done in several ways. I myself made use of roboflow
    - These are the labels I used: vehicle, person, traffic_light_red, traffic_light_orange, traffic_light_green, traffic_sign_30, traffic_sign_60, traffic_sign_90, bike, motobike
    - You can choose the labels you use when annotating, all code is dynamic so it will work with different labels.

### Lane detection data
1) In "CARLA_0.9.13/PythonAPI/Research-Project-CARLA/ Carla-Lane-Detection-Dataset-Generation/src/config.py" you can set how much data you want to collect per run and for which folder.
2) Open a terminal and go to "CARLA_0.9.13/PythonAPI/Research-Project-CARLA/Carla-Lane-Detection-Dataset-Generation/src"
3) Run fast_lane_detection.py
    - This creates numpy files in the "data/raws" folder and a JSON file with the coordinates in the "data/dataset" folder.
4) Run dataset_generator.py
    - This creates the images based on the numpy files. The images can be found in the "data/debug" folder.
5) Now we need to convert the coordinates from the JSON file to a .txt file.
    - cd back to "CARLA_0.9.13/PythonAPI/Research-Project-CARLA"
    - mkdir a ‘labels’ folder.
    - Run convert_json_to_txt.py
    - If necessary, adjust the path based on what folder you are using in the simulator.(default Town01_Opt)

## Train
### YoloV5
After collecting the data, we can now start training models ourselves.
1) Go to "CARLA_0.9.13/PythonAPI/Research-Project- CARLA/train/yolo/yolov5/models/yolov5s.yaml"
2) Open the file and on line 4, adjust the number of classes to the number of classes you have.(if you used my classes this is number 10.)
3) Place the folder with data in yolov5 format in the yolov5 folder.(see installation guide to download the data)
4) Go in a terminal to “CARLA_0.9.13/PythonAPI/Research-Project-CARLA/train/yolo/yolov5” and run ```python train.py --img 416 --batch 16 --epochs 100 --data ./FOLDER_CONTAINING_DATA/data.yaml --cfg ./models/yolov5s.yaml --weights '' --name yolov5s_results --cache``
5) It is recommended to create a wandb account in order to easily receive logs, wandb support is implemented in the yolo training script.
    - Alternatively, tensorboard can be used. The model can now be found in the logs folder: "runs/train/yolov5s_resultsX/weights/best.pt"
    - If you want to use this model then move the file to "CARLA_0.9.13/PythonAPI/Research-Project-CARLA/models"

### DETR
1) Go to “CARLA_0.9.13/PythonAPI/Research-Project-CARLA/train/detr-pytorch”
2) Place the data in the "content" folder manually or via roboflow in the notebook.
3) Run train_detr.ipynb
4) All further info can be found in the notebook.

### LSTR
For training the LSTR model, I used Azure.
1) Go to “CARLA_0.9.13/PythonAPI/Research-Project-CARLA/train/lstr_azure”
2) Create an .env file based on env_example.txt and enter your azure credentials.
3) Run train.py
4) After training, the model can be found on azure in the outputs folder.
5) Download the model and place it in "CARLA_0.9.13/PythonAPI/Research-Project-CARLA/models"

## Simulation
### Models
If you have trained models yourself you must first change the paths to the correct models in the "models" folder, you can do this in the drive_pygame.py file on lines 224 & 225.

### Start simulation
1) Open a terminal and go to CARLA_0.9.13/PythonAPI/Research-Project-CARLA”
2) Run drive_pygame.py with the desired models as parameters.
    - Parameter --lane. Pick lane detection model.
      - Ht => Houghtransform model
      - Lstr=> LSTR model
    - Parameter --object. Pick object detection model.
      - Yolo=> You Only Look Once model.
      - Detr=> DETR model.
    - Parameter --segmentation. Pick segmentation model.
      - Segform=> Gebruik het segform model.
      - IMPORTANT: This parameter is not combinable with the lane and object parameter.
3) example: ```python drive_pygame.py -l lstr -o yolo```
    - Starts the simulation with the lstr model for lane detection and the yolo model for object detection. 

## Possible problems
### Script does not run due to timeout error
Two common reasons for this error are:
1) The CARLA simulator did not start
    - The scripts and simulation can only work if the CARLA simulator is started.
2) CARLA simulator runs but on the wrong port.
    - Be sure that port 2000 is free for CARLA. If you cannot free port 2000 then you must modify the port in the code to your chosen port.
    - It is possible that due to a problem that CARLA is no longer responding on port 2000. Completely shutting down CARLA via task manager and restarting may fix this problem.
### Script does not run due to import error
Be sure to check that your scripts are running from the anaconda envoirement.
### YOLO throws a dataset not found error
1) Go to the folder where your data is stored.
2) Open de data.yaml file.
3) Check if the paths of train/test/val are correct.
      
# Acknowledgements
[DETR](https://github.com/facebookresearch/detr)

[LSTR](https://github.com/liuruijin17/LSTR)

[SegFormer](https://github.com/NVlabs/SegFormer)

[YoloV5](https://github.com/ultralytics/yolov5)

[Carla-Lane-Detection-Dataset-Generation](https://github.com/Glutamat42/Carla-Lane-Detection-Dataset-Generation)
