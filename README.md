<h1>World model-based Proprioception-only Navigation
and Locomotion over Obstacles</h1>
   
## Requirements
1. Create a new python virtual env with python 3.6, 3.7 or 3.8 (3.8 recommended)
2. Install pytorch:
    - `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117`
3. Install Isaac Gym
    - Download and install Isaac Gym Preview 3 (Preview 2 will not work!) from https://developer.nvidia.com/isaac-gym
    - `cd isaacgym/python && pip install -e .`
4. Install other packages:
    - `sudo apt-get install build-essential --fix-missing`
    - `sudo apt-get install ninja-build`
    - `pip install setuptools==59.5.0`
    - `pip install ruamel_yaml==0.17.4`
    - `sudo apt install libgl1-mesa-glx -y`
    - `pip install opencv-contrib-python`
    - `pip install -r requirements.txt`

## Training
```
python legged_gym/scripts/train.py --task=go2_baseline --headless
```

## Visualization
**Please make sure you have trained the WMP before**
```
python legged_gym/scripts/play.py --task=go2_baseline
```


## Acknowledgments

We thank the authors of the following projects for making their code open source:

- [leggedgym](https://github.com/leggedrobotics/legged_gym)
- [dreamerv3-torch](https://github.com/NM512/dreamerv3-torch)
- [AMP_for_hardware](https://github.com/Alescontrela/AMP_for_hardware)
- [parkour](https://github.com/ZiwenZhuang/parkour/tree/main)
- [extreme-parkour](https://github.com/chengxuxin/extreme-parkour)
