# AgriRobot
Plant disease detection system using pre-trained model "MobileNetV2".

This is a system for potatoes leaf disease detection using CNN, This system is a 6-wheeled robotic car with a Raspberry pi 4B board and a camera, connected to Wi-Fi to be controlled via computer or phone.
The 6 wheels of the car are powered by 12V 320RPM DC gear motors, and two BTS7960B 43A PWM DC motor drivers, the three motors on the right are connected to a motor driver and the other three motors on the left are connected to the other motor driver.
As for the camera, it is placed on an anti-shake PT Pan/Tilt camera platform that is moved by servo motors, one to move up and down, and the other to move right and left.

we used the GPIO of the Raspberry pi 4B card to connecting the system components to the Raspberry pi 4 board.
for the CNN model we use the pre-trained model "MobileNetV2", and to train this model we use the datset from kaggle from this link "https://www.kaggle.com/datasets/rizwan123456789/potato-disease-leaf-datasetpld/data".
to control this robot we use the ssh connection via @IP between my computer and the raspberry pi 4 card. and we control it via a we interface.
