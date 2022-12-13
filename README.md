# CD-ACS Model for Breast Cancer area detection in the field of Computational Pathology

# Table of repo Contents
1. [Environment Setup](#environment_setup)
2. [Dataset Setup](#dataset_setup)
3. [Training](#training)
4. [Inferencing](#inferencing)
5. [Additional tools support](#tools)
    1. [jupyterlab-nvdashboard](#tool_jupyterlab_nvdashboard)

<br/><br/>

## Environment Setup <div id='environment_setup'></div>

- Previously used hardware specs:
```
CPU: AMD Ryzen 7 5800
RAM: 64 GB
GPU: EVGA NVIDIA GeForce RTX 3080
uname : Linux 5.11.0-49-generic #55-Ubuntu SMP Wed Jan 12 17:36:34 UTC 2022 x86_64 x86_64 x86_64 GNU/Linux
Docker: Docker version 20.10.7, build 20.10.7-0ubuntu5~21.04.2
Model architecture: TensorFlow 2.10.0-dev20220427
```

### Docker image

- Using docker to build the image
``` bash
docker build -t tf . --no-cache
```

- After building, run the image by using the command from below and replace some fields with `""` quotation marks :
``` bash
docker run \
    -d \
    --name tf \
    --gpus all \
    --privileged \
    -p 8888:8888 \
    -p 6006:6006 \
    -v "Repository path ":/tf/volume \
    tf \
    jupyter lab --ip='*' --NotebookApp.token='' --NotebookApp.password='' --allow-root --no-browser
```

- After successfully running the container, open JupyterLab by entering `localhost:8888`. (FYI, port 6006 is for TensorBoard)

<br/>

### Docker compose
- Or simply run with docker-compose
``` bash
docker-compose -f docker_env/docker-compose.yml up -d
```

<br/><br/>

## Dataset Setup <div id='dataset_setup'></div>

We're using `tfds` and `Google drive` as our dataset manager and where the repo is stored.
- To create the dataset, prepare 3 types of images:
1. Original images in jpg/png format
2. Tissue-level masks with jpg/png format
3. Nucleus-level masks with jpg/png format
- After preparation, put train/val original/tissue-level images into the folder named image/target respectively, and put test original/tissue-level images into the folder named test-iamge/test-target respectively. Finally, put all Nucleus-level masks into the folder named CR.
- Zip it into a `.zip` file and upload it to Google Drive with a shared link. The folder structure is as below:
```
- dataset.zip
    - CR
    - image
    - target
    - test-image
    - test-target
```
- Retrieve the dataset shared URL and extract the id of the file, e.g. `https://drive.google.com/file/d/0B4FkkLP2MB8kMDRiNDkxOGEtMDhmNC00NzJjLThkNzQtZDc0MDNlNWVhZjk1/view?usp=sharing&resourcekey=0-k2XR2jyQonn-pTD8ndivgA` extracted to `0B4FkkLP2MB8kMDRiNDkxOGEtMDhmNC00NzJjLThkNzQtZDc0MDNlNWVhZjk1`. And replace `shared-url-here` with id in `datasets/camelyon16/camelyon16.py`
- To build the dataset, we `cd` into one of the datasets. (Under docker container environment)
```
cd datasets/camelyon16
tfds build
```

- After successfully building datasets which you can confirm by seeing terminal logs presented messages like `Dataset generation complete...` and printed out `tfds.core.DatasetInfo` etc.

## Training <div id='training'></div>

In the jupyter server, open `CDACS_model_training.ipynb` file to start training.

`*Remarks: Since we faced a problem with very large image inputs, we programmed specific modules for process large images batch-ly and subsequently.` Details are in `dataset_utils.py`, `patch_large_image.py`, and `model_utils.py`.

<br/><br/>

## Inferencing <div id='inferencing'></div>

Inferencing large images using modules inside the repo.

In the jupyter server, open the `CDACS_model_inferencing.ipynb` file to start inferencing original datasets (H&E and Fluorescence).

<br/><br/>

## Additional tools support <div id='tools'></div>

### [jupyterlab-nvdashboard](https://github.com/rapidsai/jupyterlab-nvdashboard) <div id='tool_jupyterlab_nvdashboard'></div>
- Realtime monitoring of GPU utilization
- Realtime monitoring of Machine utilization (CPU, Memory, Disk Network, etc.)

![Side panel](https://i.imgur.com/jrdDv3I.png)
![Realtime monitoring](https://i.imgur.com/BnotDwy.png)