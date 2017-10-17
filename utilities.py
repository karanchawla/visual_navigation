
# coding: utf-8

# In[1]:
try:
    # for Python2
    from Tkinter import *   ## notice capitalized T in Tkinter 
except ImportError:
    # for Python3
    from tkinter import *   ## notice lowercase 't' in tkinter here

import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
import pandas as pd
import pickle
import glob
print("OpenCV Version : %s " % cv2.__version__)


# In[3]:


def show_image(loc, title, img, width, open_new_window = False):
    if open_new_window:
        plt.figure(figsize=(width, width))
    plt.subplot(*loc)
    plt.title(title, fontsize=8)
    plt.axis('off')
    if len(img.shape) == 3:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray')
    if open_new_window:
        plt.show()
        plt.close()


# In[4]:


def preprocess_color(image):
    cropped_img = image[70:137, :, :]
    cropped_img_normalized = cropped_img/255 - 0.5
    return cropped_img_normalized


# In[5]:


def preprocess_grayscale(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return preprocess_color(gray_img)


# In[6]:


def laplacian_operator(img):
    cropped_img = img[70:137, :, :]
    laplacian = np.zeros_like(cropped_img)
    laplacian[:, :, 0] = np.absolute(cv2.Laplacian(cropped_img[:, :, 0], cv2.CV_64F))
    if debug:
        show_image((1, 1, 1), "laplacian 0", laplacian[:, :, 0], 1)
    laplacian[:, :, 1] = np.absolute(cv2.Laplacian(image_matrix_cropped[:, :, 1], cv2.CV_64F))
    if debug:
        show_image((1, 1, 1), "laplacian 0", laplacian[:, :, 1], 1)
    laplacian[:, :, 2] = np.absolute(cv2.Laplacian(image_matrix_cropped[:, :, 2], cv2.CV_64F))
    if debug:
        show_image((1, 1, 1), "laplacian 0", laplacian[:, :, 2], 1)
    laplacian_max = np.amax(laplacian, 2)
    laplacian_norm = laplacian/255 - 0.5
    return laplacian_norm


# In[7]:


def randomize_dataset_csv(csv_path):
    driving_log = pd.read_csv(csv_path, header=None)
    driving_log = driving_log.sample(frac=1).reset_index(drop=True)
    print("Overwriting CSV file: ", csv_path)
    driving_log.to_csv(csv_path, header=None, index=False)
    print("Done.")


# In[8]:


def get_driving_log_path(image_input_dir):
    assert (os.path.exists(image_input_dir))
    log_file_list = glob.glob(os.path.join(image_input_dir, '*.csv'))
    assert (len(log_file_list) == 1)
    log_path = log_file_list[0]
    return log_path


# In[9]:


def get_dataset_from_csv(image_input_dir):
    log_path = get_driving_log_path(image_input_dir)
    print("Reading from CSV log file", log_path)
    driving_log = pd.read_csv(log_path, header=None)
    return driving_log


# In[10]:


def get_dataset_from_pickle(pickle_file_path):
    with open(pickle_file_path, mode='rb') as f:
        pickle_data = pickle.load(f)
    return pickle_data


# In[1]:


def batch_preprocess(image_input_dir, l_r_correction=0.2, debug=False, measurement_range=None):
    driving_log = get_dataset_from_csv(image_input_dir)
    if measurement_range[0]:
        measurement_index = measurement_range[0]
    else:
        measurement_index = 0
    
    if measurement_range[1]:
        max_measurement_index = measurement_range[1]
    else:
        max_measurement_index = driving_log.shape[0]
    
    assert(measurement_index < max_measurement_index)
    num_measurements = max_measurement_index - measurement_index
    num_images = num_measurements * 6
    y_train = np.zeros(num_images)
    X_train = np.zeros((num_images, 67, 320, 3))
    while measurement_index < max_measurement_index:
        datum_index = (measurement_index - measurement_range[0])*6
        
        #center image
        y_train[datum_index] = driving_log.iloc[measurement_index, 3]
        center_image_filename = driving_log.iloc[measurement_index, 0]
        center_img_path = os.path.join(image_input_dir, center_image_filename)
        if debug:
            print("Using center imahe path", center_img_path)
        center_image_matrix = cv2.imread(center_img_path)
        preprocessed_center_image_matrix = preprocess_color(center_image_matrix)
        X_train[datum_index, :, :, :] = preprocessed_center_image_matrix
        
        #left image
        y_train[datum_index + 1] = driving_log.iloc[measurement_index, 3] + l_r_correction  # left image steering value added to dataset
        left_image_filename = driving_log.iloc[measurement_index, 1]
        left_image_path = os.path.join(image_input_dir, left_image_filename)
        if debug:
            print("Using left image path", left_image_path)
        left_image_matrix = cv2.imread(left_image_path)
        preprocessed_left_image_matrix = preprocess_color(left_image_matrix)
        X_train[datum_index + 1, :, :, :] = preprocessed_left_image_matrix  # left image matrix added to dataset
        # RIGHT CAMERA IMAGE
        y_train[datum_index + 2] = driving_log.iloc[measurement_index, 3] - l_r_correction  # right image steering value added to dataset
        right_image_filename = driving_log.iloc[measurement_index, 2]
        right_image_path = os.path.join(image_input_dir, right_image_filename)
        if debug:
            print("Using right image path", right_image_path)
        right_image_matrix = cv2.imread(right_image_path)
        preprocessed_right_image_matrix = preprocess_color(right_image_matrix)
        X_train[datum_index + 2, :, :, :] = preprocessed_right_image_matrix  # right image matrix added to dataset
        
        # flip center image
        flipped_center = cv2.flip(preprocessed_center_image_matrix, flipCode=1)
        y_train[datum_index + 3] = y_train[datum_index]*-1
        X_train[datum_index + 3, :, :, :] = flipped_center
        # flip left image
        flipped_left = cv2.flip(preprocessed_left_image_matrix, flipCode=1)
        y_train[datum_index + 4] = y_train[datum_index + 1]*-1
        X_train[datum_index + 4, :, :, :] = flipped_left
        # flip right image
        flipped_right = cv2.flip(preprocessed_right_image_matrix, flipCode=1)
        y_train[datum_index + 5] = y_train[datum_index + 2]*-1
        X_train[datum_index + 5, :, :, :] = flipped_right
        measurement_index += 1
        
        if debug:
            plt.figure(figsize=(15, 5))
            show_image((2, 3, 1), "Left View w/ Steering Angle " + str(y_train[datum_index]) + " Degrees", cv2.cvtColor(cv2.convertScaleAbs(preprocessed_left_image_matrix + 0.5, alpha=255), cv2.COLOR_BGR2RGB))
            show_image((2, 3, 2), "Center View w/ Steering Angle " + str(y_train[datum_index]) + " Degrees", cv2.cvtColor(cv2.convertScaleAbs(preprocessed_center_image_matrix + 0.5, alpha=255), cv2.COLOR_BGR2RGB))
            show_image((2, 3, 3), "Right View w/ Steering Angle " + str(y_train[datum_index + 2]) + " Degrees", cv2.cvtColor(cv2.convertScaleAbs(preprocessed_right_image_matrix + 0.5, alpha=255), cv2.COLOR_BGR2RGB))
            show_image((2, 3, 4), "Flipped Left View w/ Steering Angle " + str(y_train[datum_index + 4]) + " Degrees", cv2.cvtColor(cv2.convertScaleAbs(flipped_left + 0.5, alpha=255), cv2.COLOR_BGR2RGB))
            show_image((2, 3, 5), "Flipped Center View w/ Steering Angle " + str(y_train[datum_index + 3]) + " Degrees", cv2.cvtColor(cv2.convertScaleAbs(flipped_center + 0.5, alpha=255), cv2.COLOR_BGR2RGB))
            show_image((2, 3, 6), "Flipped Right View w/ Steering Angle " + str(y_train[datum_index + 5]) + " Degrees", cv2.cvtColor(cv2.convertScaleAbs(flipped_right + 0.5, alpha=255), cv2.COLOR_BGR2RGB))
            plt.show()
            plt.close()
        print('Pre-processed ', measurement_index, ' of ', max_measurement_index, ' measurements. Images:', center_image_filename, ' ', left_image_filename, ' ', right_image_filename)
        
        preprocessed_dataset = {'features': X_train, 'labels': y_train}
    return preprocessed_dataset


# In[4]:


def save_dict_to_pickle(dataset, file_path):
    print("Saving data to", file_path, "...")
    pickle.dump(dataset, open(file_path, "wb"), protocol = 4) #protocol=4 allows file sizes > 4GB
    print("Done")

