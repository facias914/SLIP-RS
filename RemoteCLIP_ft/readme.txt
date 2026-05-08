### 环境安装
    conda create -n gaze python==3.10 -y
    conda activate gaze
    pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
    pip install -r requirements.txt


### 训练
    1.训练数据集准备，结构如下：
        /image/
        ├── ship/
        │   ├── Aircraft Carrier/
        │   │       Aircraft Carrier_00001.png
        │   │       ...
        │   └── Amphibious Assault/
        │           Amphibious Assault_00001.png
        │           ...
        ├── plane/
        │   ├── Bomber/
        │   │       Bomber_00001.png
        │   │       ...
        │   └── Chartered/
        │           Chartered_00001.png
        │           ...
        ├── Oiltank/
        │   ├── Oiltank_00001.png
        │   ├── Oiltank_00002.png
        │   └── ...

        /label/
        ├── ship/
        │   ├── Aircraft Carrier.json
        │   └── Amphibious Assault.json
        ├── plane/
        │   ├── Bomber.json
        │   └── Chartered.json
        └── Oiltank.json