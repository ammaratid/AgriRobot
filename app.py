from flask import Flask, render_template, request, Response
from picamera2 import Picamera2
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import RPi.GPIO as GPIO
import time
#import threading

app = Flask(__name__)

# Initialize GPIO
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)  # Mode de numérotation BCM

# Motor Driver Pins
RPWM_RIGHT = 18
LPWM_RIGHT = 19
RPWM_LEFT = 20
LPWM_LEFT = 21
R_EN = 22  # Right Enable
L_EN = 23  # Left Enable

# Servo Pins
SERVO_UP_DOWN = 24
SERVO_LEFT_RIGHT = 25

# Initial setup function
def initialize_gpio():
    # Motor pins setup
    GPIO.setup(RPWM_RIGHT, GPIO.OUT)
    GPIO.setup(LPWM_RIGHT, GPIO.OUT)
    GPIO.setup(RPWM_LEFT, GPIO.OUT)
    GPIO.setup(LPWM_LEFT, GPIO.OUT)
    GPIO.setup(R_EN, GPIO.OUT)
    GPIO.setup(L_EN, GPIO.OUT)

    # Servo pins setup
    GPIO.setup(SERVO_UP_DOWN, GPIO.OUT)
    GPIO.setup(SERVO_LEFT_RIGHT, GPIO.OUT)
    # Call the initialization function

initialize_gpio()

# Setup PWM for motors and servos
pwm_rpwm_right = GPIO.PWM(RPWM_RIGHT, 100)  # 100 Hz frequency for motor
pwm_lpwm_right = GPIO.PWM(LPWM_RIGHT, 100)
pwm_rpwm_left = GPIO.PWM(RPWM_LEFT, 100)
pwm_lpwm_left = GPIO.PWM(LPWM_LEFT, 100)

servo_up_down = GPIO.PWM(SERVO_UP_DOWN, 50)  # 50 Hz frequency for servo
servo_left_right = GPIO.PWM(SERVO_LEFT_RIGHT, 50)

# Start PWM at 0 duty cycle to initialize
pwm_rpwm_right.start(0)
pwm_lpwm_right.start(0)
pwm_rpwm_left.start(0)
pwm_lpwm_left.start(0)
servo_up_down.start(0)
servo_left_right.start(0)

# Load the trained model
model = torch.jit.load('quantized_model_scripted.pt')
model.eval()
device = torch.device('cpu')

# Transformation for input data
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Adjust based on your model input size
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Use same normalization as dur
])

# Initialize the camera
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'RGB888', "size": (1920, 1080)}))
picam2.set_controls({"AfMode": 2})
picam2.start()
# Helper function for model inference
def predict_disease(frame):
    img = Image.fromarray(frame)
    img = transform(img).unsqueeze(0)  # Apply transformations and add batch dimension
    with torch.no_grad():
        img = img.to(device)
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)

    # Map the prediction to class names
    class_names = ['Early Blight', 'Healthy', 'Late Blight']
    return class_names[predicted.item()]

# Video stream generator
def generate_video():
    while True:
        frame = picam2.capture_array()

        # Resize the frame to 800x600 using Bicubic interpolation
        frame_resized = cv2.resize(frame, (800, 600), interpolation=cv2.INTER_CUBIC)
        # frame is in RGB format
        prediction = predict_disease(frame_resized)

        # Convert RGB to BGR (OpenCV uses BGR)
        frame_resized = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Add prediction text to the frame
        cv2.putText(frame_resized, f'Disease: {prediction}', (250, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', frame_resized)
        if not ret:
            continue
        frame = buffer.tobytes()
        # Yield the frame in a format compatible with Flask
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')
# Endpoint for the video stream
@app.route('/video_feed')
def video_feed():
    return Response(generate_video(), mimetype='multipart/x-mixed-replace; boundary=frame')
# Control endpoint for movement and servo
@app.route('/move', methods=['POST'])
def move():
    try:
        action = request.form.get('action')
        speed = int(request.form.get('speed', 50))  # Get speed, default to 50 if not provided
    except KeyError as e:
        return f"Missing parameter: {str(e)}", 400
    except ValueError:
        return "Invalid speed value", 400

    # Log received actions and speed
    print(f"Received action: {action}, Speed: {speed}")

    # Execute the appropriate motor/servo action
    if action == 'forward':
        move_forward(speed)
    elif action == 'backward':
        move_backward(speed)
    elif action == 'left':
        turn_left(speed)
    elif action == 'right':
        turn_right(speed)
    elif action == 'stop':
        stop_motors()
    elif action == 'servo_up':
        servo_up()
    elif action == 'servo_down':
        servo_down()
    elif action == 'servo_left':
        servo_left()
    elif action == 'servo_right':
        servo_right()
    else:
        return "Invalid action", 400

    return 'OK'
# Motor control functions
def move_forward(speed):
    GPIO.output(R_EN, GPIO.HIGH)
    GPIO.output(L_EN, GPIO.HIGH)
    pwm_rpwm_right.ChangeDutyCycle(speed)
    pwm_lpwm_right.ChangeDutyCycle(0)
    pwm_rpwm_left.ChangeDutyCycle(speed)
    pwm_lpwm_left.ChangeDutyCycle(0)

def move_backward(speed):
    GPIO.output(R_EN, GPIO.HIGH)
    GPIO.output(L_EN, GPIO.HIGH)
    pwm_rpwm_right.ChangeDutyCycle(0)
    pwm_lpwm_right.ChangeDutyCycle(speed)
    pwm_rpwm_left.ChangeDutyCycle(0)
    pwm_lpwm_left.ChangeDutyCycle(speed)

def turn_right(speed):
    GPIO.output(R_EN, GPIO.HIGH)
    GPIO.output(L_EN, GPIO.HIGH)
    pwm_rpwm_right.ChangeDutyCycle(0)
    pwm_lpwm_right.ChangeDutyCycle(0)
    pwm_rpwm_left.ChangeDutyCycle(speed)
    pwm_lpwm_left.ChangeDutyCycle(0)

def turn_left(speed):
    GPIO.output(R_EN, GPIO.HIGH)
    GPIO.output(L_EN, GPIO.HIGH)
    pwm_rpwm_right.ChangeDutyCycle(speed)
    pwm_lpwm_right.ChangeDutyCycle(0)
    pwm_lpwm_left.ChangeDutyCycle(0)

def stop_motors():
    pwm_rpwm_right.ChangeDutyCycle(0)
    pwm_lpwm_right.ChangeDutyCycle(0)
    pwm_rpwm_left.ChangeDutyCycle(0)
    pwm_lpwm_left.ChangeDutyCycle(0)
# Servo control functions
def servo_up():
    servo_up_down.ChangeDutyCycle(7)  # Move the servo up
    time.sleep(0.01)  # Wait for the servo to reach the position
    servo_up_down.ChangeDutyCycle(0)  # Stop sending signal

def servo_down():
    servo_up_down.ChangeDutyCycle(12)  # Move the servo down
    time.sleep(0.01)  # Wait for the servo to reach the position
    servo_up_down.ChangeDutyCycle(0)  # Stop sending signal

def servo_left():
    servo_left_right.ChangeDutyCycle(7)  # Move the servo left
    time.sleep(0.01)  # Wait for the servo to reach the position
    servo_left_right.ChangeDutyCycle(0)  # Stop sending signal

def servo_right():
    servo_left_right.ChangeDutyCycle(12)  # Move the servo right
    time.sleep(0.01)  # Wait for the servo to reach the position
    servo_left_right.ChangeDutyCycle(0)  # Stop sending signal

# GPIO cleanup
def cleanup():
    stop_motors()
    servo_up_down.stop()
    servo_left_right.stop()
    GPIO.cleanup()

# Call cleanup when the program ends
try:
    # Start the Flask app
    app.run(host='0.0.0.0', port=5000)
finally:
    cleanup()