# dm-face-detection
Simulation driver monitoring by use face detection to get eye tracking with dlib and mediapipe solution API.  

Need to donload pre-trained face dector model [68_face_landmarks](https://jumpshare.com/v/fFozZRTtoeHbkyShnEVl) when run **dms_example.py**, and **dlib_face_detection.py** code.  

## Install  

**Conda virtual env**  
```bash

conda create --name [env_name]  python=3.8

conda activate [env_name]

conda install -c conda-forge dlib

pip install opencv-python scipy imutils pyglet mediapipe
```
