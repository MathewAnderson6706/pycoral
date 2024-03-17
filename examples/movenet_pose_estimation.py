# Lint as: python3
# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""Example using PyCoral to estimate a single human pose with Edge TPU MoveNet.

To run this code, you must attach an Edge TPU to the host and
install the Edge TPU runtime (`libedgetpu.so`) and `tflite_runtime`. For
device setup instructions, see coral.ai/docs/setup.

For more details about MoveNet and its best practices, please see
https://www.tensorflow.org/hub/tutorials/movenet

Example usage:
```
bash examples/install_requirements.sh movenet_pose_estimation.py

python3 examples/movenet_pose_estimation.py \
  --model test_data/movenet_single_pose_lightning_ptq_edgetpu.tflite  \
  --input test_data/squat.bmp
```
"""


# The original code does pose detection of an image
# This fork does is instead of an image, we use a webcam to do VideoCapture and do PoseDetection there
# And instead of capturing all 17 points, we just capture 3 points(Wrist, Elbow, Shoulder)
# Also added is a bicep counter where printed on the frame is a counter for how many curls the user does
# Includes FPS and Right Wrist Location as well
from PIL import Image
from PIL import ImageDraw
from pycoral.adapters import common
from pycoral.utils.edgetpu import make_interpreter
import cv2 as cv
import numpy as np
import time

_NUM_KEYPOINTS = 17

def calculate_angle(wrist, elbow, shoulder):
    vc1 = wrist - elbow
    vc2 = shoulder - elbow
    dot_product = np.dot(vc1, vc2)
    norm_product = np.linalg.norm(vc1) * np.linalg.norm(vc2)
    angle_rad = np.arccos(dot_product / norm_product)
    angle_deg = np.degrees(angle_rad)
    return angle_deg


def main():
  # output='movenet_result.jpg'
  model = 'test_data/movenet_single_pose_lightning_ptq_edgetpu.tflite'
  interpreter = make_interpreter(model)
  interpreter.allocate_tensors()
  
  cap = cv.VideoCapture(0)
  t, fps = 0, 0
  pTime = 0
  count = 0
  curling = False
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
    
    # Saving the specific values of the 3 we need
    r_wrist = pose[10]
    r_elbow = pose[8]
    r_shoulder = pose[6]
    
    # Calculating using the method of the 3 points
    angle_at_elbow = calculate_angle(r_wrist[:2], r_elbow[:2], r_shoulder[:2])
    
    # Using the value to see if user is curling or not
    # To add to the count
    if angle_at_elbow < 30 and not curling:  
        count += 1
        curling = True 
    elif angle_at_elbow > 90:
        curling = False
    
    # Draw keypoints on the frame for only 3 points
    draw = ImageDraw.Draw(img)
    width, height = img.size
    keypoint_indices = [6, 8, 10]
    for i in keypoint_indices:
        draw.ellipse(
            xy=[
                pose[i][1] * width - 2, pose[i][0] * height - 2,
                pose[i][1] * width + 2, pose[i][0] * height + 2
            ],
            fill=(255, 0, 0))
    
    # Convert back to OpenCV format and display the frame
    frame_with_keypoints = np.array(img)
    
    # Calculate the location of the right wrist
    r_wrist_x = int(pose[10][1] * width)
    r_wrist_y = int(pose[10][0] * height)

    # Display the location of the right wrist on the frame
    cv.putText(frame_with_keypoints, f'Right Wrist: ({r_wrist_x}, {r_wrist_y})', (width - 400, 30), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    
    # Display Right Bicep Curls on the frame
    cv.putText(frame_with_keypoints, f'Right Bicep Curls: {count}', (width - 400, 60), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

    
    
    # FPS
    if t == 5:
        cTime = time.time()
        fps = 5 / (cTime - pTime)
        pTime = cTime
        t = 1
    else:
        t = t + 1
    cv.putText(frame_with_keypoints, f'FPS: {int(fps)}', (20, 30), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)
    
    
    
    
    cv.imshow('Frame', frame_with_keypoints)
    
    
    # Press 'q' quit
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

  cap.release()
  cv.destroyAllWindows()


if __name__ == '__main__':
  main()
