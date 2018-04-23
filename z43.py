import numpy as np
import cv2
#import cv
from scipy.misc import imresize
from moviepy.editor import VideoFileClip
from IPython.display import HTML
from keras.models import load_model
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cProfile
import time
import sys
import math
######################################
import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
import csv
######################################
model_path='./model/model.ckpt'

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.' \
                                                            '  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # Define the name of the tensors
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    # Get the needed layers' outputs for building FCN-VGG16
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    image_input = tf.get_default_graph().get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = tf.get_default_graph().get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3 = tf.get_default_graph().get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4 = tf.get_default_graph().get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7 = tf.get_default_graph().get_tensor_by_name(vgg_layer7_out_tensor_name)

    return image_input, keep_prob, layer3, layer4, layer7
def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """


    weights_regularized_l2 = 1e-4

    # 1x1 Convolution to preserve spatial information.
    enc_layer7 = tf.layers.conv2d(
        vgg_layer7_out, num_classes, kernel_size=1,
        kernel_initializer= tf.random_normal_initializer(stddev=0.01),
        kernel_regularizer= tf.contrib.layers.l2_regularizer(weights_regularized_l2), name='enc_layer7')

    enc_layer4 = tf.layers.conv2d(
        vgg_layer4_out, num_classes, kernel_size=1,
        kernel_initializer= tf.random_normal_initializer(stddev=0.01),
        kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-4), name='enc_layer4')
    enc_layer3 = tf.layers.conv2d(
        vgg_layer3_out, num_classes, kernel_size=1,
        kernel_initializer= tf.random_normal_initializer(stddev=0.01),
        kernel_regularizer= tf.contrib.layers.l2_regularizer(weights_regularized_l2), name='enc_layer3')


######################################
#  Upsample
    dec_layer1 = tf.layers.conv2d_transpose(
        enc_layer7, num_classes, kernel_size=4, strides=(2, 2),
        padding='same',
        kernel_initializer= tf.random_normal_initializer(stddev=0.01),
        kernel_regularizer= tf.contrib.layers.l2_regularizer(weights_regularized_l2), name='dec_layer1')

    # Skip connection from the vgg_layer4_out
    dec_layer2 = tf.add(
        dec_layer1, enc_layer4, name='dec_layer2')

    # Deconvolution: Make shape the same as layer3
    dec_layer3 = tf.layers.conv2d_transpose(
        dec_layer2, num_classes, kernel_size=4, strides=(2, 2),
        padding='same',
        kernel_initializer= tf.random_normal_initializer(stddev=0.01),
        kernel_regularizer= tf.contrib.layers.l2_regularizer(weights_regularized_l2), name='dec_layer3')

    # Same for layer4.
    dec_layer4 = tf.add(
        dec_layer3, enc_layer3, name='dec_layer4')
    decoder_output = tf.layers.conv2d_transpose(
        dec_layer4, num_classes, kernel_size=16, strides=(8, 8),
        padding='same',
        kernel_initializer= tf.random_normal_initializer(stddev=0.01),
        kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-4), name='dec_layer4')

    return decoder_output
########################################################################
def predict_images2(test_data_path, print_speed=False):
    num_classes = 2
    image_shape = (160, 576)
    runs_dir = './runs'

    # Path to vgg model
    vgg_path = os.path.join('./data', 'vgg')

    with tf.Session() as sess:
        # Predict the logits
        input_image, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, vgg_path)
        nn_last_layer = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)
        logits = tf.reshape(nn_last_layer, (-1, num_classes))

        # Restore the saved model
        saver = tf.train.Saver()
        saver.restore(sess, model_path)
        print("Load saved Model in file: %s" % model_path)

        # Predict the samples
        helper.pred_samples(runs_dir, test_data_path, sess, image_shape, logits, keep_prob, input_image, print_speed)

########################################################################
def predict_images2(test_data_path, print_speed=False):
    num_classes = 2
    image_shape = (160, 576)
    runs_dir = './runs'

    # Path to vgg model
    vgg_path = os.path.join('./data', 'vgg')

    with tf.Session() as sess:
        # Predict the logits
        input_image, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, vgg_path)
        nn_last_layer = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)
        logits = tf.reshape(nn_last_layer, (-1, num_classes))

        # Restore the saved model
        saver = tf.train.Saver()
        saver.restore(sess, model_path)
        print("Load saved Model in file: %s" % model_path)

        # Predict the samples
        helper.pred_samples(runs_dir, test_data_path, sess, image_shape, logits, keep_prob, input_image, print_speed)

########################################################################
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
# Load Keras model
model = load_model('AVO_FCN_model_2.h5')
try:
  os.rename("/home/stack/PROJECTS/LANE0/output.avi", "/home/stack/PROJECTS/LANE0/output_LAST.avi")
except Exception, e:
  print ('EXCEPT:  rename = ', e)
try:
  os.rename("/home/stack/PROJECTS/LANE0/output_after.avi", "/home/stack/PROJECTS/LANE0/output_after_LAST.avi")
except Exception, e:
  print ('EXCEPT:  rename = ', e)
DEBUG_ON = True
DO_ONCE = True
IMG_CNT = 0

WRITE_IMG = True
#WRITE_IMG = False
# Global parameters

# Gaussian smoothing
kernel_size = 3

# Canny Edge Detector
low_threshold = 1
#low_threshold = 10
#low_threshold = 20
high_threshold = 150

# Region-of-interest vertices
trap_bottom_width = 0.85  
trap_top_width = 0.07  
trap_height = 0.4  

# Hough Transform
rho = 2 
theta = 1 * np.pi/180 
#threshold = 50     
#threshold = 10     
#min_line_length = 10
#min_line_length = 20
#min_line_length = 30
#max_line_gap = 20
#max_line_gap = 50

#test_pic = './IMAGES1/fliph_test6730.jpg'
#img = cv2.imread(test_pic)
#mimg = mpimg.imread(test_pic)
#gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#edges = cv2.Canny(gray,50,150,apertureSize = 3)
#lines = cv2.HoughLines(edges,1,np.pi/180,200)
#plt.imshow(mimg)
#plt.show()
#for rho,theta in lines[0]:
#    a = np.cos(theta)
#    b = np.sin(theta)
#    x0 = a*rho
#    y0 = b*rho
#    x1 = int(x0 + 1000*(-b))
#    y1 = int(y0 + 1000*(a))
#    x2 = int(x0 - 1000*(-b))
#    y2 = int(y0 - 1000*(a))

#    cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
#img_out = cv2.line(img, (100,100), (300,300), (0,0,255),4)
#cv2.imwrite('lane_lines_out.jpg',img)

###  TIMEIT
def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print ('%r  %2.2f ms' % \
                  (method.__name__, (te - ts) * 1000))
        return result
    return timed

###  BEGIN LANE LINE 
def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_noise(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

############################################

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    ignore_mask_color = (0,255,0)
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def region_of_interest2(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
#    mask = np.zeros_like(img)   
    mask = img.copy()
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    ignore_mask_color = (0,255,0)
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
#    masked_image = cv2.bitwise_and(img, mask)

#    masked_image = cv2.addWeighted(mask, 0.2, img, Beta, Lambda)
#    masked_image = cv2.addWeighted(mask, 0.2, img, .8, Lambda)
    masked_image = cv2.addWeighted(mask, 0.2, img, Beta, Lambda)
    return masked_image
############################################
def extend_point(x1, y1, x2, y2, length):
    """ Takes line endpoints and extroplates new endpoint by a specfic length"""
    line_len = np.sqrt((x1 - x2)**2 + (y1 - y2)**2) 
    x = x2 + (x2 - x1) / line_len * length
    y = y2 + (y2 - y1) / line_len * length
    return x, y
  
def reject_outliers(data, cutoff, thresh=0.08):
    """Reduces jitter by rejecting lines based on a hard cutoff range and outlier slope """
    data = np.array(data)
    data = data[(data[:, 4] >= cutoff[0]) & (data[:, 4] <= cutoff[1])]
    m = np.mean(data[:, 4], axis=0)
    return data[(data[:, 4] <= m+thresh) & (data[:, 4] >= m-thresh)]


def merge_lines(lines):
    """Merges all Hough lines by the mean of each endpoint, 
       then extends them off across the image"""
    
    lines = np.array(lines)[:, :4] ## Drop last column (slope)
    
    x1,y1,x2,y2 = np.mean(lines, axis=0)
    x1e, y1e = extend_point(x1,y1,x2,y2, -1000) # bottom point
    x2e, y2e = extend_point(x1,y1,x2,y2, 1000)  # top point
    line = np.array([[x1e,y1e,x2e,y2e]])
    
    return np.array([line], dtype=np.int32)

############################################
def draw_lines2(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


#def draw_lines(img, lines, color=[255, 0, 0], thickness=10):
def draw_lines(img, lines, color=[0, 0, 255], thickness=10):
	"""
	NOTE: this is the function you might want to use as a starting point once you want to 
	average/extrapolate the line segments you detect to map out the full
	extent of the lane (going from the result shown in raw-lines-example.mp4
	to that shown in P1_example.mp4).  
	
	Think about things like separating line segments by their 
	slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
	line vs. the right line.  Then, you can average the position of each of 
	the lines and extrapolate to the top and bottom of the lane.
	
	This function draws `lines` with `color` and `thickness`.	
	Lines are drawn on the image inplace (mutates the image).
	If you want to make the lines semi-transparent, think about combining
	this function with the weighted_img() function below
	"""
	# In case of error, don't draw the line(s)
	if lines is None:
		return
	if len(lines) == 0:
		return
	draw_right = True
	draw_left = True
	
	# Find slopes of all lines
	# But only care about lines where abs(slope) > slope_threshold
	slope_threshold = 0.5
	slopes = []
	new_lines = []
	for line in lines:
                try:
		  x1, y1, x2, y2 = line[0]  # line = [[x1, y1, x2, y2]]
                except:
                  print ('None:  UNPACK x1, y1, x2, y2 = line[0]  # line = [[x1, y1, x2, y2]]' )
                  return

		
		# Calculate slope
		if x2 - x1 == 0.:  # corner case, avoiding division by 0
			slope = 999.  # practically infinite slope
		else:
			slope = (y2 - y1) / (x2 - x1)
			
		# Filter lines based on slope
		if abs(slope) > slope_threshold:
			slopes.append(slope)
			new_lines.append(line)
		
	lines = new_lines
	
	# Split lines into right_lines and left_lines, representing the right and left lane lines
	# Right/left lane lines must have positive/negative slope, and be on the right/left half of the image
	right_lines = []
	left_lines = []
	for i, line in enumerate(lines):
		x1, y1, x2, y2 = line[0]
		img_x_center = img.shape[1] / 2  # x coordinate of center of image
		if slopes[i] > 0 and x1 > img_x_center and x2 > img_x_center:
			right_lines.append(line)
		elif slopes[i] < 0 and x1 < img_x_center and x2 < img_x_center:
			left_lines.append(line)
			
	# Run linear regression to find best fit line for right and left lane lines
	# Right lane lines
	right_lines_x = []
	right_lines_y = []
	
	for line in right_lines:
		x1, y1, x2, y2 = line[0]
		
		right_lines_x.append(x1)
		right_lines_x.append(x2)
		
		right_lines_y.append(y1)
		right_lines_y.append(y2)
		
	if len(right_lines_x) > 0:
		right_m, right_b = np.polyfit(right_lines_x, right_lines_y, 1)  # y = m*x + b
	else:
		right_m, right_b = 1, 1
		draw_right = False
		
	# Left lane lines
	left_lines_x = []
	left_lines_y = []
	
	for line in left_lines:
		x1, y1, x2, y2 = line[0]
		
		left_lines_x.append(x1)
		left_lines_x.append(x2)
		
		left_lines_y.append(y1)
		left_lines_y.append(y2)
		
	if len(left_lines_x) > 0:
		left_m, left_b = np.polyfit(left_lines_x, left_lines_y, 1)  # y = m*x + b
	else:
		left_m, left_b = 1, 1
		draw_left = False
	
	# Find 2 end points for right and left lines, used for drawing the line
	# y = m*x + b --> x = (y - b)/m
	y1 = img.shape[0]
	y2 = img.shape[0] * (1 - trap_height)
	
	right_x1 = (y1 - right_b) / right_m
	right_x2 = (y2 - right_b) / right_m
	
	left_x1 = (y1 - left_b) / left_m
	left_x2 = (y2 - left_b) / left_m
	
	# Convert calculated end points from float to int
	y1 = int(y1)
	y2 = int(y2)
	right_x1 = int(right_x1)
	right_x2 = int(right_x2)
	left_x1 = int(left_x1)
	left_x2 = int(left_x2)
	
	# Draw the right and left lines on image
	if draw_right:
		cv2.line(img, (right_x1, y1), (right_x2, y2), color, thickness)
	if draw_left:
		cv2.line(img, (left_x1, y1), (left_x2, y2), color, thickness)
            
#	if draw_right:
#		cv2.line(img, (right_x1, y1), (right_x2, y2+1000), color, thickness)
#	if draw_left:
#		cv2.line(img, (left_x1, y1), (left_x2, y2+1000), color, thickness)
  ##########################################################
       

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros(img.shape, dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, Alpha=0.8, Beta=1., Lambda=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * Alpha + img * Beta + Lambda
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, Alpha, img, Beta, Lambda)

def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


################################

def remove_outliers(slopes, m = 2):
    med = np.mean(slopes)
    stand_dev = np.std(slopes)
    for slope in slopes:
        if abs(slope - med) > (m * stand_dev):
            slopes.remove(slope)
        if slope == 0:
            slopes.remove(slope)
    return slopes

def lane_lines(img):
    global DEBUG_ON
    #Gray-scale
    gray = grayscale(img)
    
    #Smooth it a bit with Gaussian Blur
    kernel_size = 9
    blur_gray = gaussian_blur(gray, kernel_size)
    
    #Canny Edge 
    low_threshold = 1
#    low_threshold = 10
#    low_threshold = 20
    high_threshold = 180
    edges = canny(blur_gray, low_threshold, high_threshold)
    
    # Masking
    imshape = img.shape
    vertices = np.array([[(0,imshape[0]),(450, 325), (550, 325), (imshape[1],imshape[0])]], dtype=np.int32)  
    masked_edges = region_of_interest(edges, vertices)

    # Hough transform 
    rho = 5 
    theta = np.pi/30 
#    threshold = 20     
    threshold = 1
#    threshold = 10     
#    threshold = 5     
#    threshold = 30     
#    threshold = 50     
#    min_line_len = 10 
#    min_line_len = 20 
#    min_line_len = 30 
#    max_line_gap = 25    
#    max_line_gap = 50
#    line_image = np.copy(img)*0 

    lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]), min_line_len, max_line_gap)
    
    try:
      if lines is not None:
        print ('Lines = ', lines)
    except:
        print (' ------------ Lines = None ')
        return


    left_lines = []
    right_lines = []
    right_slopes = []
    left_slopes = []
    try:
	    for line in lines:
	        for x1,y1,x2,y2 in line:
	            if (x2 - x1 != 0):
	              slope = (y2-y1)/(x2-x1)
	            else:
	              slope = 999
	            intercept = y1 - slope*x1
	            line_length = np.sqrt((y2-y1)**2+(x2-x1)**2)

	            if DEBUG_ON:
	              if (np.isnan(x1)):
	                print ('x1', x1)
	              if (np.isnan(x2)):
	                print ('x2', x2)
	              if (np.isnan(y1)):
	                print ('y1', y1)
	              if (np.isnan(y1)):
	                print ('y2', y2)


	            if DEBUG_ON:
	              print ('y2 = ', y2)
	              print ('y1 = ', y1)
	              print ('x2 = ', x2)
	              print ('x1 = ', x1)



	            if slope == 0:
	              print ('SLOPE = ',slope)
	              slope = 0.0001
#              break

	            if slope < 0 and line_length > 10:
	                left_lines.append(line)
	                left_slopes.append(slope)
	            elif slope > 0 and line_length > 10:
	                right_lines.append(line)
	                right_slopes.append(slope)
    except:    
      print ('No lane lines ')
      return 
    #Average line 
    avg_left_pos = [sum(col)/len(col) for col in zip(*left_lines) if len(col) != 0]
    avg_right_pos = [sum(col)/len(col) for col in zip(*right_lines) if len(col) != 0]
    
    avg_left_slope = np.mean(remove_outliers(left_slopes))
    avg_right_slope = np.mean(remove_outliers(right_slopes))
    
    avg_left_line = []
    for x1,y1,x2,y2 in avg_left_pos:
        x = int(np.mean([x1, x2])) 
        y = int(np.mean([y1, y2])) 
        slope = avg_left_slope
        if slope == 0:
           print ('SLOPE = 0')
           slope = 0.1
        b = -(slope * x) + y 



        avg_left_line = [int((325-b)/slope), 325, int((539-b)/slope), 539] #Line for the image 
    
    avg_right_line = []
    for x1,y1,x2,y2 in avg_right_pos:
        x = int(np.mean([x1, x2]))
        y = int(np.mean([y1, y2]))
        slope = avg_right_slope
        if slope == 0:
           print ('SLOPE = 0')
           slope = 0.1
        b = -(slope * x) + y
        avg_right_line = [int((325-b)/slope), 325, int((539-b)/slope), 539]
    
    lines = [[avg_left_line], [avg_right_line]]
    
    draw_lines(line_image, lines)

    line_edges = weighted_img(line_image, img)
    
    return line_edges



###  END LANE LINE 
# Class to average lanes with
class Lanes():
    def __init__(self):
        self.recent_fit = []
        self.avg_fit = []
        self.last_fit = []

def reject_outliers_1(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]

def reject_outliers_2(data, m = 2.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    return data[s<m]
##############################################################
# Nessesary variables
Alpha = 0.9                 # Weight factor for initial image
Beta = 0.4                 # Weight factor for new image
Lambda = 0.22                # Scalar added to each sum (of new and initial image), see weighted_img function
kernel_size = 7         # Size of the n x n matrix used as kernel in gaussian blur
low_threshold = 50      # Value for the canny function, defining the first threshold
high_threshold = 150    # Value for the canny function, defining the second threshold
hist_frames = 10        # History of how many frames back to remember lane lines
rho = 1                 # Distance resolution in pixels of the Hough grid
theta = np.pi/90        # Angular resolution in radians of the Hough grid
#threshold = 15          # Minimum number of votes (intersections in Hough grid cell)
threshold = 15          # Minimum number of votes (intersections in Hough grid cell)
min_line_length = 70    # Minimum number of pixels in a line
#min_line_length = 10    # Minimum number of pixels in a line
max_line_gap = 180      # Maximum gap in pixels between connectable line segments
min_slope_value = 0.4   # Defining the minimum slope value. Lines under this value is not lane lines.
left_line_stack = []    # For keeping a stack of left lines
right_line_stack = []   # For keeping a stack of right lines
left_line_history = []  # For keeping a history of left lines, if there is no left lines
right_line_history = [] # For keeping a history of right lines, if there is no right lines
#H = img.shape[0]        # Getting the height of the image
#Hr = H*0.6              # Reducing the height
#W = img.shape[1]        # Getting the width of the image
ly = np.array([20, 100, 100], dtype = "uint8") # Low lalue for yellow HSV.
uy = np.array([30, 255, 255], dtype = "uint8") # Hig value for yellow HSV.
#vertices = np.array([(x * W, y * H) for (x,y) in [[0.05,1], [0.46, 0.60], [0.54, 0.60], [1,1]]], np.int32) # ROI
##############################################################
# Pipeline
def process_image(img):
    rho = 1                 # Distance resolution in pixels of the Hough grid
    theta = np.pi/90        # Angular resolution in radians of the Hough grid
    min_line_length = 70    # Minimum number of pixels in a line
    ly = np.array([20, 100, 100], dtype = "uint8") # Low lalue for yellow HSV.
    uy = np.array([30, 255, 255], dtype = "uint8") # Hig value for yellow HSV.
    kernel_size = 7         # Size of the n x n matrix used as kernel in gaussian blur

    # Converts the image to gray
    gray_img = grayscale(img)
    
    # Making an HSV (Hue, Saturation, Value) image
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # Masking the HSV values
    mask_yellow = cv2.inRange(img_hsv, ly, uy)
    mask_white = cv2.inRange(gray_img, 200, 255)
    mask_yw = cv2.bitwise_or(mask_white, mask_yellow)
    mask_yw_img = cv2.bitwise_and(gray_img, mask_yw)
    
    # Bluring the image with a low pass filter called "Gaussian blur"
    blur_img = gaussian_blur(mask_yw_img, kernel_size)

    # Detecting edges with the Canny function
    edge_img = canny(mask_yw_img, low_threshold, high_threshold)
    
    # Applying Region of Interest (ROI)
    maskd_img = region_of_interest(edge_img, [vertices])


    ####  RESIZE NP ARRAY
#    small_img = imresize(image, (80, 160, 3))
    
    print ('maskd_img.shape =', maskd_img.shape)
    # Coloring the lane
    try:
       lane_img = hough_lines(maskd_img, rho, theta, threshold, min_line_length, max_line_gap)
       print ('TRY lane_img = hough_lines  ')
       print ('maskd_img.shape =', maskd_img.shape)
    except Exception,e:
       print ('EXCEPT lane_img = hough_lines  error' , e)
       print ('maskd_img.shape =', maskd_img.shape)
    
    # Displaying img with lane lanes on top of original image
#    print ('BEFORE final_img:  lane_img, img  ',lane_img, img)

    ####  RESIZE NP ARRAY
#    small_img = imresize(image, (80, 160, 3))
#    lane_img_resized = imresize(lane_img, (480,640,3))
#    gray = cv2.cvtColor(image_resized,cv2.COLOR_BGR2GRAY)
#    lane_img_resized = cv2.cvtColor(lane_img, cv2.CV_GRAY2RGB)
#     stacked_img = np.stack((img,)*3, -1)
    lane_img_resized = np.stack((lane_img,)*3, -1)
#    lane_img_resized = imresize(lane_img, (640,480,3))
#    lane_img_resized = imresize(lane_img, (480,640,3))

#        img_resized = cv2.resize(image_rgb, desired_dim, interpolation=cv2.INTER_LINEAR)
#    lane_img_resized = cv2.resize(lane_img, (640,480), interpolation=cv2.INTER_LINEAR)

    print ('AFTER lane_img RESIZE but BEFORE final_img:  lane_img_resized, lane_img  ',lane_img_resized.shape, lane_img.shape)
#    print ('AFTER lane_img RESIZE but BEFORE final_img:  lane_img, img  ',lane_img, img)
#    final_img = weighted_img(lane_img, img, Alpha, Beta, Lambda)
    try:
       print ('TRY BEFORE final_img:  lane_img_resized.shape, img.shape  ',lane_img_resized.shape, img.shape)
       final_img = weighted_img(lane_img_resized, img, Alpha, Beta, Lambda)
       print ('TRY AFTER final_img:  lane_img_resized.shape, img.shape  ',lane_img_resized.shape, img.shape)
       if (COUNTER % 10 == 0):
          cv2.imwrite('maskd.jpg', maskd_img)
          cv2.imwrite('edge_img.jpg', edge_img)
          cv2.imwrite('houghlines.jpg', lane_img)
    except:
       print ('EXCEPT final_img:  lane_img_resized.shape, img.shape  ',lane_img_resized.shape, img.shape)
       print ('EXCEPT final_img:  lane_img  ',lane_img.shape[0], lane_img.shape[1])
       print ('EXCEPT final_img:  img  ',img.shape[0], img.shape[1])
##############################################################
#    image1 = cv2.imread('img.png')
    image1 = img
#    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # Converts the image to gray
    gray_img2 = grayscale(img)

    # Making an HSV (Hue, Saturation, Value) image
    img_hsv2 = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # Masking the HSV values
    mask_yellow2 = cv2.inRange(img_hsv2, ly, uy)
    mask_white2 = cv2.inRange(gray_img2, 200, 255)
    mask_yw2 = cv2.bitwise_or(mask_white2, mask_yellow2)
    mask_yw_img2 = cv2.bitwise_and(gray_img2, mask_yw2)

    # Bluring the image with a low pass filter called "Gaussian blur"
    blur_img2 = gaussian_blur(mask_yw_img2, kernel_size)

    # Detecting edges with the Canny function
    edge_img2 = canny(mask_yw_img2, low_threshold, high_threshold)


    # Applying Region of Interest (ROI)
    maskd_img2 = region_of_interest(edge_img2, [vertices])


    gray=cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
    dst = cv2.Canny(gray, 50, 200)


    try:
       lane_img2 = hough_lines(maskd_img2, rho, theta, threshold, min_line_length, max_line_gap)
#       lines= cv2.HoughLinesP(dst, 1, math.pi/180.0, 210, np.array([]), 0, 0)
#       lines= cv2.HoughLinesP(maskd_img2, 1, math.pi/180.0, 10, np.array([]), 0, 0)
       lines= cv2.HoughLinesP(dst, 1, math.pi/180.0, 10, np.array([]), 0, 0)
       print ('TRY lane_img 2 = hough_lines  ', lane_img2.shape)
       print ('maskd_img2.shape =', maskd_img2.shape)
    except Exception,e:
       print ('EXCEPT lane_img = hough_lines  error' , e)
       print ('maskd_img2.shape =', maskd_img2.shape)

#    lines= cv2.HoughLinesP(dst, 1, math.pi/180.0, 210, np.array([]), 0, 0)


    lines= cv2.HoughLinesP(dst, 1, math.pi/180.0, 210, np.array([]), 120, 5)
#    lines= cv2.HoughLinesP(dst, 1, math.pi/180.0, 210, np.array([]), 0, 0)

#    lines= cv2.HoughLinesP(dst, 1, math.pi/180.0, 150, np.array([]), 0, 0)
#    lines= cv2.HoughLinesP(dst, 1, math.pi/180.0, 100, np.array([]), 0, 0)
#    lines= cv2.HoughLinesP(dst, 1, math.pi/180.0, 200, np.array([]), 0, 0)
#    lines= cv2.HoughLinesP(dst, 1, math.pi/180.0, 210, np.array([]), 0, 0)
#    lines= cv2.HoughLinesP(dst, 1, math.pi/180.0, 230, np.array([]), 0, 0)
#    lines= cv2.HoughLinesP(dst, 1, math.pi/180.0, 250, np.array([]), 0, 0)
#    lines= cv2.HoughLinesP(dst, 1, math.pi/180.0, 200, np.array([]), 0, 0)
#    lines= cv2.HoughLinesP(dst, 1, math.pi/180.0, 100, np.array([]), 0, 0)
#    lines= cv2.HoughLines(dst, 1, math.pi/180.0, 100, np.array([]), 0, 0)
#    a,b,c = lines.shape
    try:
       print ('TRY BEFORE final_img 2')
#       final_img2 = weighted_img(lane_img_resized, img, Alpha, Beta, Lambda)
#       final_img2 = weighted_img(lines, img, Alpha, Beta, Lambda)
       final_img2 = weighted_img(lane_img2, img, Alpha, Beta, Lambda)
       print ('TRY AFTER final_img 2: ') 
       if (COUNTER % 10 == 0):
          cv2.imwrite('maskd_img2.jpg', maskd_img2)
          cv2.imwrite('edge_img2.jpg', edge_img2)
          cv2.imwrite('lane_img2.jpg', lane_img2)
          cv2.imwrite('final_img2.jpg', final_img2)
    except Exception,e:
       print ('EXCEPT lane_img2  error ===' , e)
       print ('EXCEPT lane_img2.shape ' , lane_img2.shape)
##############################################################
    try:
        a,b,c = lines.shape
    	for i in range(a):
	        rho = lines[i][0][0]
	        theta = lines[i][0][1]
	        a = math.cos(theta)
	        b = math.sin(theta)
	        x0, y0 = a*rho, b*rho
	        pt1 = ( int(x0+1000*(-b)), int(y0+1000*(a)) )
	        pt2 = ( int(x0-1000*(-b)), int(y0-1000*(a)) )
	        cv2.line(image1, pt1, pt2, (0, 0, 255), 2, cv2.LINE_AA)     
#        cv2.imshow(image1)
	        if (COUNTER % 10 == 0):
	          cv2.imwrite('my_hough_lines.jpg',image1)
    except Exception,e:
        print ('EXCEPT:  lines.shape', e) 
##############################################################
#    final_img = img


    final_img = image1
#    final_img = final_img2

    return final_img
##############################################################
#@timeit
def seg_lane_lines(image):
    """ Takes in a road image, re-sizes for the model,
    predicts the lane to be drawn from the model in G color,
    recreates an RGB image of a lane and merges with the
    original road image.
    """
    global DO_ONCE
    global IMG_CNT 

#    if DO_ONCE:
#      print ('image type is', type(image))
#      DO_ONCE = False

    # Get image ready for feeding into model
    small_img = imresize(image, (80, 160, 3))
    small_img = np.array(small_img)
    small_img = small_img[None,:,:,:]

    # Make prediction with neural network (un-normalize value by multiplying by 255)
    prediction = model.predict(small_img)[0] * 255

#    prediction = reject_outliers_2(model.predict(small_img)[0] * 255)
    prediction2 = reject_outliers_2(prediction, m=2)
#    print ("Prediction = ", prediction)
#    print ("Prediction = ", prediction[10])
#    print ("Prediction2 = ", prediction2[10])

    # Add lane prediction to list for averaging
    lanes.recent_fit.append(prediction)
#    try:
#      lanes.recent_fit.append(prediction2)
#    except:
#      print ("#########   No lanes recent fit")
    # Only using last five for average
    if len(lanes.recent_fit) > 5:
#        lanes.recent_fit = reject_outliers_2(lanes.recent_fit[1:], m=2)
        lanes.recent_fit = lanes.recent_fit[1:]
#        print ("Prediction = ", prediction[0])
#        print ("Prediction2 = ", prediction2[0])
#        print ("lanes.recent_fit = ", lanes.recent_fit)

    # Calculate average detection
    lanes.avg_fit = np.mean(np.array([i for i in lanes.recent_fit]), axis = 0)

    # Generate fake R & B color dimensions, stack with G
    blanks = np.zeros_like(lanes.avg_fit).astype(np.uint8)
    lane_drawn = np.dstack((blanks, lanes.avg_fit, blanks))

    # Re-size to match the original image
    lane_image = imresize(lane_drawn, (720, 1280, 3))

    image_resized = imresize(image,(720,1280,3))



    # Merge the lane drawing onto the original image
#    result = cv2.addWeighted(image, 1, lane_image, 1, 0)
#    img_as_array = np.asarray(image,dtype="int32")
#    result = cv2.addWeighted(img_as_array, 1, lane_image, 1, 0)

#gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#edges = cv2.Canny(gray,50,150,apertureSize = 3)
#lines = cv2.HoughLines(edges,1,np.pi/180,200)
    gray = cv2.cvtColor(image_resized,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,50,150,apertureSize = 3)
    lines = cv2.HoughLines(edges,1,np.pi/180,200)

    lines_as_nparr = np.asarray(lines)
    edges_as_nparr = np.asarray(edges)
#    lines_as_bgr = cv2.cvtColor(lines_as_nparr,cv2.COLOR_GRAY2BGR)
#    edges_as_bgr = cv2.cvtColor(lines,cv2.COLOR_GRAY2BGR)

#    for rho,theta in lines[0]:
#        a = np.cos(theta)
#        b = np.sin(theta)
#        x0 = a*rho
#        y0 = b*rho
#        x1 = int(x0 + 1000*(-b))
#        y1 = int(y0 + 1000*(a))
#        x2 = int(x0 - 1000*(-b))
#        y2 = int(y0 - 1000*(a))
#
#        cv2.line(image_resized,(x1,y1),(x2,y2),(0,0,255),2)

#    cv2.imwrite('houghlines3.jpg',image_resized)
    cv2.imwrite('cannyedge3.jpg',edges)


#    result_lines = cv2.addWeighted(image_resized, 1, edges_as_bgr, 1, 0)
#    result_lines = cv2.addWeighted(image_resized, 1, lines_as_bgr, 1, 0)
#    result_lines = cv2.addWeighted(image_resized, 1, lines_as_nparr, 1, 0)
#    result_lines = cv2.addWeighted(image_resized, 1, lines, 1, 0)
#    result = cv2.addWeighted(result_lines, 1, lane_image, 1, 0)
#    result = cv2.addWeighted(lines_as_bgr, 1, lane_image, 1, 0)


#    plt.imshow(edges,cmap='gray')
#    plt.imshow(lines[0])
#    plt.show()

     

    result = cv2.addWeighted(image_resized, 1, lane_image, 1, 0)


    imshape = image.shape
    vertices = np.array([[(205,imshape[0]),(277, 307), (317, 307), (400,imshape[0])]], dtype=np.int32)
    output_roi = region_of_interest2(image, vertices)

#    cv2.imshow('edges',edges)
    cv2.imshow('output_roi',output_roi)

    output_roi_resized = imresize(output_roi,(720,1280,3))

    try:
#    	result2 = cv2.addWeighted(output_roi_resized, 0.2, lane_image, 1, 0)
    	result2 = cv2.addWeighted(output_roi_resized, 0.2, lane_image, 1, 0)
    except Exception, e:
	print ('EXCEPT addWeight output_roi', e)
    cv2.imshow('result2',result2)

    
#    result = cv2.addWeighted(image_resized, 1, edges_as_bgr, 1, 0)
#    result = cv2.addWeighted(image_resized, 1, lines_as_bgr, 1, 0)

#    img_out = cv2.line(img_as_array, (100,100), (300,300), (0,0,255),4)
#    cv2.imshow('test', img_out)
#    filename = 'test' + str(IMG_CNT) + '.jpg'
#    IMG_CNT += 1
#    if WRITE_IMG:
#      cv2.imwrite(filename, img_out)
#    result = image
#    result = np.asarray(img_out)
#    result = img_out

#    result = edges
#    result = lines

    imshape = image.shape
    vertices = np.array([[(205,imshape[0]),(277, 307), (317, 307), (400,imshape[0])]], dtype=np.int32)  
    outputroi = region_of_interest2(edges, vertices)

#    cv2.imshow('edges',edges)
    cv2.imshow('outputroi',outputroi)
    return result

lanes = Lanes()

#########################################
#cap = cv2.VideoCapture(0)

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


#test_pic = './IMAGES1/fliph_test6730.jpg'
#img = cv2.imread(test_pic)
#mimg = mpimg.imread(test_pic)
#gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#edges = cv2.Canny(gray,50,150,apertureSize = 3)
#lines = cv2.HoughLines(edges,1,np.pi/180,200)
#plt.imshow(lane_lines(mimg))
#plt.show()


#test_pic = './IMAGES1/fliph_test6730.jpg'
#img = cv2.imread(test_pic)
#img = frame
#mimg = mpimg.imread(test_pic)
#gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#edges = cv2.Canny(gray,50,150,apertureSize = 3)
#lines = cv2.HoughLines(edges,1,np.pi/180,10)
#for rho,theta in lines[0]:
#    a = np.cos(theta)
#    b = np.sin(theta)
#    x0 = a*rho
#    y0 = b*rho
#    x1 = int(x0 + 1000*(-b))
#    y1 = int(y0 + 1000*(a))
#    x2 = int(x0 - 1000*(-b))
#    y2 = int(y0 - 1000*(a))

#    img_out = cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
#img_out = cv2.line(img, (100,100), (300,300), (0,0,255),4)
#cv2.imwrite('lane_lines_out.jpg',img)
#img_out = cv2.line(mimg,(x1,y1),(x2,y2),(0,0,255),2)

#cv2.imshow('img_out',img_out)
#plt.imshow(img_out)
#plt.show()
##############################################


cap = cv2.VideoCapture(0)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))
#out_after = cv2.VideoWriter('output_after.avi',fourcc, 20.0, (640,480))
out_after = cv2.VideoWriter('output_after.avi',fourcc, 20.0, (1280,720))
out2_after = cv2.VideoWriter('output2_after.avi',fourcc, 20.0, (1280,720))
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
COUNTER = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        frame = cv2.flip(frame,0)

        # write the flipped frame
        out.write(frame)

#        cv2.imshow('frame',frame)

######  BEGIN VIDEO PROCESSING PIPELINE

        img = frame

        H = img.shape[0]        # Getting the height of the image
#        Hr = H*0.6              # Reducing the height
        Hr = H*0.99              # Reducing the height
        W = img.shape[1]        # Getting the width of the image
#        vertices = np.array([(x * W, y * H) for (x,y) in [[0.05,1], [0.46, 0.60], [0.54, 0.60], [1,1]]], np.int32) # ROI
#        vertices = np.array([[(x*0.,y),(x*.475, y*.575), (x*.525, y*.575), (x,y)]], dtype=np.int32)
#        vertices = np.array([(x * W, y * H) for (x,y) in [[0.475,0.575], [0.525, 0.575], [0.525, 0.575], [1,1]]], np.int32) # ROI
        rows, cols = img.shape[:2]
        bottom_left  = [cols*0.1, rows*0.95]
        top_left     = [cols*0.4, rows*0.6]
        bottom_right = [cols*0.9, rows*0.95]
        top_right    = [cols*0.6, rows*0.6] 
        vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
##############################################################

#    img_out = cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
#        output = lane_lines(frame)
#       	framef = cv2.flip(frame,0)
        framef = frame
       	framef = cv2.flip(frame,-1)
       	framef = cv2.flip(frame,0)
#      	framef = cv2.flip(framef,1)
#        out_after.write(framef)
        output = seg_lane_lines(framef)
#        out_after.write(output)
#        out_after.write(output)
        try:
        	output = np.array(output, dtype=np.uint8)
#        	output = cv2.flip(output,0)
#        	output = cv2.flip(output,1)
#                vertices = np.array([[(205,imshape[0]),(277, 307), (317, 307), (400,imshape[0])]], dtype=np.int32)  
        	cv2.imshow('output', output)
                out_after.write(output)
        except:
		print ('None: in show output')
####################################################

####################################################
        framef_np = np.asarray(framef)
        imshape = framef.shape
        vertices = np.array([[(205,imshape[0]),(277, 307), (317, 307), (400,imshape[0])]], dtype=np.int32)  
#        vertices = np.array([[(205,imshape[0]),(277, 307), (277, 307), (400,imshape[0])]], dtype=np.int32)  
#        vertices = np.array([[(205,imshape[0]),(227, 307), (227, 307), (400,imshape[0])]], dtype=np.int32)  
#        vertices = np.array([[(205,imshape[0]),(330, 325), (330, 325), (400,imshape[0])]], dtype=np.int32)  
#        vertices = np.array([[(205,imshape[0]),(230, 325), (450, 325), (400,imshape[0])]], dtype=np.int32)  
#        vertices = np.array([[(300,imshape[0]),(450, 225), (550, 225), (400,imshape[0])]], dtype=np.int32)  
#        vertices = np.array([[(0,imshape[0]),(450, 325), (550, 325), (imshape[1],imshape[0])]], dtype=np.int32)  
#        vertices = np.array([[(0,imshape[0]),(450, 325), (550, 325), (imshape[1],imshape[0])]], dtype=np.int32)  
        output4 = region_of_interest2(framef_np, vertices)
        try:
        	output4 = np.array(output4, dtype=np.uint8)
    		output4_sized = imresize(output4,(720,1280,3))
        	cv2.imshow('output4', output4)
                out2_after.write(output4_resized)
        except:
		print ('None: in show output4')
####################################################
        framef_np = np.asarray(framef)
        output2 = process_image(framef_np)
        try:
        	output2 = np.array(output2, dtype=np.uint8)
#        	cv2.imshow('output2', output2)
                out2_after.write(output2)
        except:
		print ('None: in show output2')
####################################################
#        framef_np = np.asarray(framef)
        
#        output3 = predict_image2(framef_np, print_speed=True)
#        try:
#        	output3 = np.array(output3, dtype=np.uint8)
#        	cv2.imshow('output3', output3)
#                out2_after.write(output3)
#        except:
#		print ('None: in show output3')
####################################################
####################################################
        framef_np = np.asarray(framef)
        imshape = framef.shape
        vertices = np.array([[(205,imshape[0]),(277, 307), (317, 307), (400,imshape[0])]], dtype=np.int32)  
#        vertices = np.array([[(205,imshape[0]),(277, 307), (277, 307), (400,imshape[0])]], dtype=np.int32)  
#        vertices = np.array([[(205,imshape[0]),(227, 307), (227, 307), (400,imshape[0])]], dtype=np.int32)  
#        vertices = np.array([[(205,imshape[0]),(330, 325), (330, 325), (400,imshape[0])]], dtype=np.int32)  
#        vertices = np.array([[(205,imshape[0]),(230, 325), (450, 325), (400,imshape[0])]], dtype=np.int32)  
#        vertices = np.array([[(300,imshape[0]),(450, 225), (550, 225), (400,imshape[0])]], dtype=np.int32)  
#        vertices = np.array([[(0,imshape[0]),(450, 325), (550, 325), (imshape[1],imshape[0])]], dtype=np.int32)  
#        vertices = np.array([[(0,imshape[0]),(450, 325), (550, 325), (imshape[1],imshape[0])]], dtype=np.int32)  
#        output4 = region_of_interest2(framef_np, vertices)
#        try:
#        	output4 = np.array(output4, dtype=np.uint8)
#        	cv2.imshow('output4', output4)
#                out2_after.write(output4)
#        except:
#		print ('None: in show output4')
####################################################
        COUNTER += 1
        print ("COUNTER = ", COUNTER)
#        if COUNTER > 100:
#          sys.exit()
######  END VIDEO PROCESSING PIPELINE

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
##############################################
