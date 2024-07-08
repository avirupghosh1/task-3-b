# Digit Checker
A basic neural network of 2 hidden layers detecting digits between (0-9)

To run the nn first run the gradient.py which creates 6 matrices which is used when running the 2nd file MAIN.py


## Download train.csv and test.csv from kaggle

scroll down and click on download all which will then extract the .zip inside nn-1 repo so that the two .py files and the two .csv files are in same directory [click here](https://www.kaggle.com/competitions/digit-recognizer/data)


## Deployment

First change the 7th parameter in the following line of gradient.py file to your actual path of nn-1 repo 

```bash
 save_parameters(w1_F, b1_F, w2_F, b2_F, w3_F, b3_F,"PATH/OF/YOUR/REPO")
```
now run the gradient.py which should give you W1.npy , W2.npy, W3.npy, B1.npy, B2.npy, B3.npy I already have some good values in there so if you donot want want to run gradient descent you can directly got to the main file.

now run the main file it should open to 2 tabs with tkinter app and image. Cross the image window to see different test examples.

if not installed before, install tkinter:-

```bash
 pip install tk
 ```
 or 
 ```bash
 pip3 install tk
 ```
- yt demo for this is [here](https://www.youtube.com/watch?v=ttxKs_6_aq8&ab_channel=AvirupGhosh)
- Now for the aruco_real create a video server using ip camera app in your phone and change the url = "path/of/your/video/stream" in the python file and run the script . You will be able to see the camera output along with aruco detection and pose estimation.
- yt video for this is demo is [here](https://www.youtube.com/watch?v=dDknjrwP-lQ&ab_channel=AvirupGhosh)
