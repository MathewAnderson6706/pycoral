from PIL import Image
from PIL import ImageDraw
from pycoral.adapters import common
from pycoral.utils.edgetpu import make_interpreter
from sense_hat import SenseHat
import cv2 as cv
import numpy as np
import time
import math
import speech_recognition as sr
import google.generativeai as genai
import os
from multiprocessing import Process, Queue
 
_NUM_KEYPOINTS = 17
 
 
def calculate_angle(wrist, elbow, shoulder):
    vec1 = wrist - elbow
    vec2 = shoulder - elbow
    dot_product = np.dot(vec1, vec2)
    norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    angle_rad = np.arccos(dot_product / norm_product)
    angle_deg = np.degrees(angle_rad)
    return angle_deg
 
def speech_recognition(queue):
    #Voice and Ai Stuff
    # Initialize the recognizer
    response = " "
    response_str = " "
    recognizer = sr.Recognizer()
    sense = SenseHat()
    # Setting up my API Key
    genai.configure(api_key='AIzaSyCSaKG2iGOTU_3SyaV1ujclVdlVHdkZyt0')
 
    # Select the AI model
    model = genai.GenerativeModel('gemini-pro')
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)
        while True:
          # Recognize speech using Google Web Speech API
          audio_data = recognizer.listen(source)
          try:
              text = recognizer.recognize_google(audio_data)
              prompt = "Identify the intent of the following text contained within curly braces. " \
                      "The intent could be one of the following: A. Read Temperature B. Read Humidity " \
                      "C. Reset Exercise Counter. Here few examples 1. if the text is {I want to start over}, " \
                      "then return 'Reset Counter' as an output 2. if the text is {Restart} then return 'Reset Counter' " \
                      "3. if the text is {How Hot is it} then return 'Read Temperature' 4. If the text is {What is the humidity} then return 'Read Humidity'. Here is the input text for your " \
                      "response: {" + text + "}.  I want the response to be what I want, in just 1 line, nothing else."
              response = model.generate_content(prompt)
              response_str = str(response.text)
              print(response_str)
              queue.put(response_str)
              #print(response.text)
          except sr.UnknownValueError:
              print("Google Web Speech API could not understand the audio")
          except sr.RequestError as e:
              print(f"Could not request results from Google Web Speech API; {e}")
 
def main():
  # output='movenet_result.jpg'
  model = 'test_data/movenet_single_pose_lightning_ptq_edgetpu.tflite'
  interpreter = make_interpreter(model)
  interpreter.allocate_tensors()
  sense = SenseHat()
  red = (255, 0, 0)
  green = (0, 255, 0)
 
  cap = cv.VideoCapture(0)
  t, fps = 0, 0
  pTime = 0
  count = 0
  count_knee = 0
  knee = False
  curling = False
 
 
  queue = Queue()
  p = Process(target=speech_recognition, args=(queue,))
  p.start()
 
  while True:
    ret, frame = cap.read()
    if not ret:
        break
 
 
    # Convert frame to PIL Image
    img = Image.fromarray(frame)
    resized_img = img.resize(common.input_size(interpreter), Image.ANTIALIAS)
    common.set_input(interpreter, resized_img)
 
    interpreter.invoke()
 
    pose = common.output_tensor(interpreter, 0).copy().reshape(_NUM_KEYPOINTS, 3)
 
    width, height = img.size
    # Saving the specific values of the 6 we need
    r_wrist = pose[10]
    l_wrist = pose[9]
    r_elbow = pose[8]
    r_shoulder = pose[6]
    r_ankle = pose[16]
    l_knee = pose[13]
 
    # Calculate the location of right anlke
    r_ankle_y = int(pose[16][0] * height)
 
    # Calculate the location of left knee
    l_knee_y = int(pose[13][0] * height)
 
    # Calculating using the method of the 3 points
    angle_at_elbow = calculate_angle(r_wrist[:2], r_elbow[:2], r_shoulder[:2])
 
    # Using the value to see if user is curling or not
    # To add to the count
    if angle_at_elbow < 30 and not curling:  
        count += 1
        count_str = str(count)
        sense.show_message(count_str, text_colour = red)
        curling = True
    elif angle_at_elbow > 120:
        curling = False
 
    if r_ankle_y > l_knee_y and not knee:
        count_knee += 1
        count_knee_str = str(count_knee)
        sense.show_message(count_knee_str, text_colour = green)
        knee = True
    elif r_ankle_y < l_knee_y:
        knee = False
  
  
    response_str = " "
    if not queue.empty():
        response_str = queue.get()

    if response_str == "Read Temperature":
        temp = sense.get_temperature()
        temp_str = "{:.1f}C".format(temp)
        sense.show_message(temp_str)
    elif response_str == "Read Humidity":
        humidity = sense.get_humidity()
        humidity_str = "{:.1f}G/KG".format(humidity)
        sense.show_message(humidity_str)
    elif response_str == "Reset Exercise Counter":
        count_knee = 0
        count = 0
        
    response_str = " "
 
    # Convert back to OpenCV format and display the frame
    frame_with_keypoints = np.array(img)
 
    # Calculate the location of the right wrist
    r_wrist_x = int(pose[10][1] * width)
    r_wrist_y = int(pose[10][0] * height)
 
    # Calculate the location of the left wrist
    l_wrist_x = int(pose[9][1] * width)
    l_wrist_y = int(pose[9][0] * height)
 
 
    distance = math.sqrt(math.pow(r_wrist_x - l_wrist_x, 2) + math.pow(l_wrist_x - l_wrist_y, 2))
 
 
 
 
 
    # if both wrists come close together, than end
    if distance < 65:
        break
 
 
 
  cap.release()
  cv.destroyAllWindows()
  p.terminate()
 
 
if __name__ == '__main__':
  main()
