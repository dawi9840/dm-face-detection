# dm-face-detection
Using face detection simulation driver monitoring to get Iris tracking with dlib and mediapipe solution API.  

Need to donload pre-trained face dector model [68_face_landmarks](https://jumpshare.com/v/fFozZRTtoeHbkyShnEVl) when run **dms_example.py**, and **dlib_face_detection.py** code.  

## Dlib

<div align="center">
<img src="https://user-images.githubusercontent.com/19554347/138228979-406de9a3-14fb-4331-bc8e-014ba5b5210b.jpg" height="150px" alt="people_face" >
<img src="https://user-images.githubusercontent.com/19554347/138228985-9008f57e-900e-4be8-8e95-9eff234efe64.png" height="150px" alt="dlib_people_face" >
</div>

## Mediapipe

<div align="center">
<img src="https://user-images.githubusercontent.com/19554347/138227437-cad71e2e-051e-4873-9fa8-c96326c9b7fa.jpg" height="150px" alt="people_face" >
<img src="https://user-images.githubusercontent.com/19554347/138227450-6512fb85-ef30-4068-b464-2e76d00519f3.png" height="150px" alt="mp_people_face" >
</div>

## Install  

**Conda virtual env**  
```bash

conda create --name [env_name]  python=3.8

conda activate [env_name]

conda install -c conda-forge dlib

pip install opencv-python scipy imutils pyglet mediapipe
```
