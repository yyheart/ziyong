# Avatar ERNeRF

## ER-NeRF : [Paper](https://arxiv.org/abs/2307.09323) | [github](https://github.com/Fictionarry/ER-NeRF.git)

## 语言: [[English](README.md)] | [简体中文]

## 安装

- 参考环境: Ubuntu18.04; CUDA11.3; CUDNN>=8.2.4, <8.7.0; gcc/g++-9;

    **安装linux库:**
    <br>
    ```shell
    sudo apt-get install libasound2-dev portaudio19-dev
    sudo apt-get install ffmpeg # 或源码编译ffmpeg
    ```
    <br>

    **源码编译openface:**
    <br>
    参考文档: `https://github.com/TadasBaltrusaitis/OpenFace.git`<br>
    openface编译完成后将`FeatureExtraction`(这是一个可执行文件), 拷贝到`/usr/local/bin`下.
    <br>
    或者将`tasks/preprocess.py`下`run_face_feature_extraction`函数中:
    ```python
    cmd = "FeatureExtraction -f {} -out_dir {}".format(input_path, temp_dir)
    ```
    这一行中`"FeatureExtraction"`修改为你的`FeatureExtraction`地址.
    <br>
    <br>

    **安装python环境:**
    <br>
    ```shell
    # 创建conda虚拟环境
    conda create -n ernerf python=3.10 -y
    conda activate ernerf

    # 安装pytorch
    pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

    # 安装pytorch3d
    pip install "git+https://github.com/facebookresearch/pytorch3d.git"

    # 安装其余依赖
    pip install -r requirements.txt
    sh install_ext.sh
    ```

- 预训练模型:

    下载预训练模型: [Google Drive](https://drive.google.com/file/d/12kz5-UwWyKzTf7z2hFUO41Jx5wnTEbJy/view?usp=drive_link)<br>
    解压并置于`<projet>/pretrained_weights`目录.
    <br>

    其中`3DMM`相关模型来自[Basel Face Model 2009](https://faces.dmi.unibas.ch/bfm/main.php?nav=1-1-0&id=details)
    <br>
    下载与转换方法如下:

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

## 使用

- 数据

    输入的视频数据应为5-10分钟左右的视频, 视频中只有一人, 且需要保证时间连续性(一镜到底).
  
    可以是这样的一段视频:<br>
    ![sample input video](./docs/sample_video.gif)

- 训练

    ```shell
    cd <PROJECT>
    export PYTHONPATH=./

    python -u create_train_task.py -i <input_video_path> --model_uid <model_name>

    # `input_video_path`是指输入视频路径.
    # `model_name`是人为赋给这段视频的UID, 方便加载对应的数据和权重.
    # `model_cfg_file`应传一个json文件路径, 如果为`None`则储存有该模特详细信息的配置文件仅会保存在`<model_dataset_folder>/<model_name>/model_data.json`中; 如果不为`None`则会在`model_cfg_file`再保存一份.
    # `preproc_only`表示仅进行该视频的"数据前处理"流程, 不会进行"训练"流程.
    # `preproc_done`表示该视频的"数据前处理"流程已经完成, 会通过直接加载"储存有该模特详细信息的配置文件"进行"训练"流程.
    # `preload`表示"训练"流程是否进行预加载.
    # 更多参数说明详见`create_train_task.py`
    ```

    训练流程分两部分:
    <br>
    - (1) 数据前处理

    会将输入视频进行人脸检测, 姿态估计, 背景抠图等流程, 详细内容见`tasks/preprocess.py`;<br>
    经过前处理后大概会得到这样一个目录结构:
    ```
    <model_dataset_folder>
    ├── <model_name>
    |	├── input_temp.mp4         # 经过格式转换后的`libx264+25fps`的视频文件
    |	├── input.mp4              # 经过抠图和替换目标背景后的视频文件
    |	├── alpha.mp4              # 经过抠图后得到的alpha视频文件
    |	├── preview.png            # 输入视频的首帧图像
    |	├── face_data.mp4          # 头像部分区域的视频文件
    |	├── head_bbox.json         # 头像部分区域的坐标(x1, y1, x2, y2)
    |	├── audio.wav              # 经过格式转换后`wav+16000sr`的音频文件
    |	├── audio_feat.npy         # 经过音频特征提取后得到的音频特征文件
    |	├── ori_imgs               # 头像视频的每一帧
    |	├── parsing_label          # 头像视频的每一帧的人脸解析结果
    |	├── parsing                # 头像视频的每一帧的人脸解析结果转换为头颈躯干
    |	├── bc.jpg                 # 头像视频中提取得到的背景图像
    |	├── gt_imgs                # ground truth图像
    |	├── torso_imgs             # 躯干部分的图像
    |	├── au.csv                 # openface FeatureExtraction得到的action units结果
    |	├── track_params.pt        # face tracking训练得到的权重文件
    |	├── transforms_train.json  # 用于训练集的transform matrix
    |	├── transforms_val.json    # 用于验证集的transform matrix
    |	├── model_data.json        # 储存有该模特详细信息的配置文件

    ```

    - (2) ER-NeRF算法训练
    
    分为头部训练, 嘴部微调, 躯干部分训练三个步骤, 详细内容见`trainer.py`;<br>

- 推理

    ```shell
    python -u create_infer_task.py -i <input_audio> -c <model_name or model_config_file>
    ```

    推理流程分为两部分:
    <br>
    - (1) 加载ER-NeRF的权重进行推理
    
    会得到推理结果, 即头部区域的图像序列, 和对应的深度图序列; <br>
    
    - (2) 推理结果后处理

    会将得到的头部区域图像序列贴回原始输入视频中的头部区域, 提供三种后处理方案, 详细内容见`tasks/postprocess.py`;<br>


## 感谢列表

- 人脸检测模型来源于: [yolov7-face](https://github.com/derronqi/yolov7-face.git)

- 人脸特征提取来源于: [OpenFace](https://github.com/TadasBaltrusaitis/OpenFace.git)

- 人脸关键点检测模型来源于: [face-alignment](https://github.com/1adrianb/face-alignment.git)

- 人脸解析模型来源于: [face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch.git)

- 人脸跟踪模型来源于: [AD-NeRF](https://github.com/YudongGuo/AD-NeRF.git)

- 人体姿态估计模型来源于: [yolov7-pose](https://github.com/trancongman276/yolov7-pose.git)

- 背景抠图模型来源于: [BackgroundMattingV2](https://github.com/PeterL1n/BackgroundMattingV2.git)

- 说话头生成模型来源于: [ER-NeRF](https://github.com/Fictionarry/ER-NeRF.git)

## 其他

- 一个微信的技术分享群, 欢迎分享和交流
<br>

![wechat](./docs/wechat_group.jpg)
