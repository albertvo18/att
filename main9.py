import numpy as np
import cv2
from scipy.misc import imresize
from moviepy.editor import VideoFileClip
from IPython.display import HTML
from keras.models import load_model
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
# Load Keras model
model = load_model('full_CNN_model.h5')

DO_ONCE = True

# Class to average lanes with
class Lanes():
    def __init__(self):
        self.recent_fit = []
        self.avg_fit = []

def road_lines(image):
    """ Takes in a road image, re-sizes for the model,
    predicts the lane to be drawn from the model in G color,
    recreates an RGB image of a lane and merges with the
    original road image.
    """
    global DO_ONCE

    if DO_ONCE:
      print ('image type is', type(image))
      DO_ONCE = False

    # Get image ready for feeding into model
    small_img = imresize(image, (80, 160, 3))
    small_img = np.array(small_img)
    small_img = small_img[None,:,:,:]

    # Make prediction with neural network (un-normalize value by multiplying by 255)
    prediction = model.predict(small_img)[0] * 255

    # Add lane prediction to list for averaging
    lanes.recent_fit.append(prediction)
    # Only using last five for average
    if len(lanes.recent_fit) > 5:
        lanes.recent_fit = lanes.recent_fit[1:]

    # Calculate average detection
    lanes.avg_fit = np.mean(np.array([i for i in lanes.recent_fit]), axis = 0)

    # Generate fake R & B color dimensions, stack with G
    blanks = np.zeros_like(lanes.avg_fit).astype(np.uint8)
    lane_drawn = np.dstack((blanks, lanes.avg_fit, blanks))

    # Re-size to match the original image
    lane_image = imresize(lane_drawn, (720, 1280, 3))

    # Merge the lane drawing onto the original image
#    result = cv2.addWeighted(image, 1, lane_image, 1, 0)
    img_as_array = np.asarray(image,dtype="int32")
#    result = cv2.addWeighted(img_as_array, 1, lane_image, 1, 0)

    result = image

    return result

lanes = Lanes()

#########################################
cap = cv2.VideoCapture(0)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))
#self._name = name + '.mp4'
#self._cap = VideoCapture(0)
#self._fourcc = VideoWriter_fourcc(*'MP4V')
#self._out = VideoWriter(self._name, self._fourcc, 20.0, (640,480))
name = 'output.mp4'
#fourcc = cv2.VideoWriter_fourcc(*'MP4V')
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#fourcc = cv2.VideoWriter_fourcc(*'X264')
#out = cv2.VideoWriter(name, fourcc, 20.0, (1280,720))
#out = cv2.VideoWriter(name, fourcc, 20.0, (640,480))

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        frame = cv2.flip(frame,0)

        # write the flipped frame
        out.write(frame)

        cv2.imshow('frame',frame)

######  BEGIN VIDEO PROCESSING PIPELINE
#        print (type(frame))
#        image = cv2.imread(frame)
#        image = mpimg.imread(frame)
#        lanes = road_lines(image)

#        plt.imshow(road_lines(image))
#        output = road_lines(frame)

#        image = cv2.imread(frame)
#        myimage = Image.fromarray(frame, 'RGB')
        myimage = Image.fromarray(frame, 'GRAY')

#        cv_img = np.array(frame / 255, dtype = np.uint8)
#        cv_img = np.array(myimage * 255, dtype = np.uint8)
#        cv_img = myimage.astype(np.uint8)
#        output = road_lines(cv_img)

        output = road_lines(myimage)
#        plt.imshow(frame)
#        plt.imshow(output)
#        plt.show()
#        print ("Finished VIDEO PROCRESSING")

######  END VIDEO PROCESSING PIPELINE

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
