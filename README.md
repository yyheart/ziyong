# Avatar ERNeRF

## ER-NeRF : [Paper](https://arxiv.org/abs/2307.09323) | [github](https://github.com/Fictionarry/ER-NeRF.git)

## Language: [English] | [[简体中文](README_CN.md)]

## 安装

- reference: Ubuntu18.04; CUDA11.3; CUDNN>=8.2.4, <8.7.0; gcc/g++-9;

    **linux packages:**
    <br>
    ```shell
    sudo apt-get install libasound2-dev portaudio19-dev
    sudo apt-get install ffmpeg # or build it from source
    ```
    <br>

    **build openface:**
    <br>
    reference doc: `https://github.com/TadasBaltrusaitis/OpenFace.git`<br>
    after build, you will get a `FeatureExtraction`, copy it to `/usr/local/bin`.
    <br>
    or you can modify `tasks/preprocess.py` function `run_face_feature_extraction` to your own `FeatureExtraction` path.
    ```python
    cmd = "<path_to_FeatureExtraction> -f {} -out_dir {}".format(input_path, temp_dir)
    ```
    <br>

    **python environment:**
    <br>
    ```shell
    conda create -n ernerf python=3.10 -y
    conda activate ernerf

    # install pytorch
    pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

    # install pytorch3d
    pip install "git+https://github.com/facebookresearch/pytorch3d.git"

    # install other dependencies
    pip install -r requirements.txt
    sh install_ext.sh
    ```

- pretrained weights:

    download from: [Google Drive](https://drive.google.com/file/d/12kz5-UwWyKzTf7z2hFUO41Jx5wnTEbJy/view?usp=drive_link)<br>
    unzip to `<projet>/pretrained_weights`.
    <br>

    the `3DMM` model is from: [Basel Face Model 2009](https://faces.dmi.unibas.ch/bfm/main.php?nav=1-1-0&id=details)
    <br>
    download and convert:

    ```shell
    wget https://github.com/YudongGuo/AD-NeRF/blob/master/data_util/face_tracking/3DMM/exp_info.npy?raw=true -O pretrained_weights/3DMM/exp_info.npy
    wget https://github.com/YudongGuo/AD-NeRF/blob/master/data_util/face_tracking/3DMM/keys_info.npy?raw=true -O pretrained_weights/3DMM/keys_info.npy
    wget https://github.com/YudongGuo/AD-NeRF/blob/master/data_util/face_tracking/3DMM/sub_mesh.obj?raw=true -O pretrained_weights/3DMM/sub_mesh.obj
    wget https://github.com/YudongGuo/AD-NeRF/blob/master/data_util/face_tracking/3DMM/topology_info.npy?raw=true -O pretrained_weights/3DMM/topology_info.npy
    ``` 

    ```shell
    # 1. 
    # copy 01_MorphableModel.mat to `<projet>pretrained_weights/3DMM`
    # 2.
    cd modules/face_tracking
    python convert_BFM.py
    ```

## Usage

- Input Data

    the input video should be 5-10 minutes long, and only one person in the video.
    <br>
    example:
    <br>
    ![sample input video](./docs/sample_video.gif)

- Training

    ```shell
    cd <PROJECT>
    export PYTHONPATH=./

    python -u create_train_task.py -i <input_video_path> --model_uid <model_name>

    # `input_video_path` means the path to the input video.
    # `model_name` means the name of the model.
    # `model_cfg_file` should be a json file path, if it's `None`, the model_config file will save to `<model_dataset_folder>/<model_name>/model_config.json`; if it's not `None`, the model_config file will save to `model_cfg_file`.
    # `preproc_only` means only run data preprocessing, and skip training.
    # `preproc_done` means whether the preprocessing has been done, if it's `True`, the preprocessing will be skipped.
    # `preload` means whether do preload data.
    # please refer to `create_train_task.py` for more details.
    ```

    - (1) Data Preprocessing

    The input video will be used for face detection, pose estimation, background matting and other processes, as detailed in `tasks/preprocess.py`
    <br>
    And you will get a directory looklike this:
    ```
    <model_dataset_folder>
    ├── <model_name>
    |	├── input_temp.mp4
    |	├── input.mp4
    |	├── alpha.mp4
    |	├── preview.png
    |	├── face_data.mp4
    |	├── head_bbox.json
    |	├── audio.wav
    |	├── audio_feat.npy
    |	├── ori_imgs
    |	├── parsing_label
    |	├── parsing
    |	├── bc.jpg
    |	├── gt_imgs
    |	├── torso_imgs
    |	├── au.csv
    |	├── track_params.pt
    |	├── transforms_train.json
    |	├── transforms_val.json
    |	├── model_data.json

    ```

    - (2) ER-NeRF Training
    
    It is divided into three steps: head training, mouth fine-tuning and torso training, please refer to `trainer.py` for more details.

- Inference

    ```shell
    python -u create_infer_task.py -i <input_audio> -c <model_name or model_config_file>
    ```

    - (1) Load ernerf weight and Inference
    
    - (2) Postprocess: Please refer to `tasks/postprocess.py` for more details.

## Acknowledgement

- Face Detection From: [yolov7-face](https://github.com/derronqi/yolov7-face.git)

- Face FeatureExtraction From: [OpenFace](https://github.com/TadasBaltrusaitis/OpenFace.git)

- Face Landmark Detection From: [face-alignment](https://github.com/1adrianb/face-alignment.git)

- Face Parsing From: [face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch.git)

- Face Tracking From: [AD-NeRF](https://github.com/YudongGuo/AD-NeRF.git)

- Pose Estimation From: [yolov7-pose](https://github.com/trancongman276/yolov7-pose.git)

- Background Matting From: [BackgroundMattingV2](https://github.com/PeterL1n/BackgroundMattingV2.git)

- Talking Head From: [ER-NeRF](https://github.com/Fictionarry/ER-NeRF.git)
