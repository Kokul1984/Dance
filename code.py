!pip install mediapipe
!wget -q https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import cv2
from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Layer, Flatten, Multiply, Add, Reshape, GlobalMaxPooling2D, Concatenate, Input, concatenate, Lambda
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
from keras.applications import ResNet50, VGG16
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
from tensorflow.keras.utils import plot_model
from keras import backend as K
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from google.colab.patches import cv2_imshow
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options,
                                       num_hands=1, min_hand_detection_confidence = 0.0)
MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

###########################################################################
##extract hand landmarks and create images ################

def draw_landmarks_on_image(rgb_image, detection_result):
  hand_landmarks_list = detection_result.hand_landmarks
  handedness_list = detection_result.handedness
  #annotated_image = np.copy(rgb_image)
  annotated_image = np.zeros_like(rgb_image)

  # Loop through the detected hands to visualize.
  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]
    handedness = handedness_list[idx]

    # Draw the hand landmarks.
    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    hand_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      hand_landmarks_proto,
      solutions.hands.HAND_CONNECTIONS,
      solutions.drawing_styles.get_default_hand_landmarks_style(),
      solutions.drawing_styles.get_default_hand_connections_style())

    # Get the top left corner of the detected hand's bounding box.
    height, width, _ = annotated_image.shape
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]
    text_x = int(min(x_coordinates) * width)
    text_y = int(min(y_coordinates) * height) - MARGIN
  return annotated_image
  
  def extract_hand_landmarks(images):
  modified_images = []
  for image in images:
    image = (image * 255).astype(np.uint8)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
    base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(base_options=base_options,
                                       num_hands=1, min_hand_detection_confidence = 0.0)
    detector = vision.HandLandmarker.create_from_options(options)
    detection_result = detector.detect(image)
    annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
    modified_images.append(annotated_image)
  images_float32 = np.zeros_like(modified_images, dtype=np.float32)
  for i in range(len(modified_images)):
      images_float32[i] = modified_images[i].astype(np.float32) / 255.0
  return np.array(images_float32)
  
  import os

# Function to process images
new_width = 224
new_height = 224
def resize_and_write(image_path,  output_path):
    try:
        # Read the image
        image = cv2.imread(image_path)

        # Resize the image
        resized_image = cv2.resize(image, (new_width, new_height))

        # Write the resized image to the output path with the same format
        cv2.imwrite(output_path, resized_image)

        print("Resized image saved to:", output_path)
    except Exception as e:
        print("Error:", e)

# Function to traverse directories
def process_directory(input_dir, output_dir):
    # Iterate over files and directories
    for root, dirs, files in os.walk(input_dir):
        # Create similar directory structure in the output directory
        rel_path = os.path.relpath(root, input_dir)
        output_subdir = os.path.join(output_dir, rel_path)
        os.makedirs(output_subdir, exist_ok=True)

        # Process each file in the current directory
        for file in files:
            input_path = os.path.join(root, file)
            output_path = os.path.join(output_subdir, file)

            # Process the image
            resize_and_write(input_path, output_path)

# Example usage
input_directory = '/content/drive/MyDrive/RESEARCH/Asamyuktha_Hastas/Testing Images'
output_directory = '/content/drive/MyDrive/RESEARCH/Dataset Small/Testing Images'
process_directory(input_directory, output_directory)

base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options,
                                       num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

# STEP 3: Load the input image.
image = mp.Image.create_from_file("/content/drive/MyDrive/RESEARCH/Pataka_001.jpg")

# STEP 4: Detect hand landmarks from the input image.
detection_result = detector.detect(image)

# STEP 5: Process the classification result. In this case, visualize it.
annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
cv2_imshow(cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
output_image_path = "/content/drive/MyDrive/RESEARCH/kokulkokul.jpg"
cv2.imwrite(output_image_path, annotated_image)

###############################################################################
##CNN models######
batch_size = 16
img_size = 224

# Set the path to your data folders
data_dir = '/content/drive/MyDrive/Research/Asamyuktha_Hastas_canveswithhand'
train_data_dir = os.path.join(data_dir, 'Training Images')
test_data_dir = os.path.join(data_dir, 'Testing Images')

# Count the number of classes and images per class in the training dataset
train_classes = os.listdir(train_data_dir)
train_num_classes = len(train_classes)
train_images_per_class = 0

for train_class in train_classes:
    train_class_path = os.path.join(train_data_dir, train_class)
    if os.path.isdir(train_class_path):
        train_images = os.listdir(train_class_path)
        train_images_per_class += len(train_images)

# Count the number of classes and images per class in the testing dataset
test_classes = os.listdir(test_data_dir)
test_num_classes = len(test_classes)
test_images_per_class = 0

for test_class in test_classes:
    test_class_path = os.path.join(test_data_dir, test_class)
    if os.path.isdir(test_class_path):
        test_images = os.listdir(test_class_path)
        test_images_per_class += len(test_images)

# Set the input image dimensions
img_width, img_height = 224, 224

train_datagen = ImageDataGenerator(
    rescale=1/255.,
    rotation_range=10,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    brightness_range=[0.9, 1.1]
)

test_datagen = ImageDataGenerator(rescale=1/255.)

train_generator = train_datagen.flow_from_directory(train_data_dir,
                                              target_size=(img_width, img_height),
                                              batch_size=batch_size,
                                              shuffle=True,
                                              class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(test_data_dir,
                                                  target_size=(img_width, img_height),
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  class_mode='categorical


class ChannelAttention(Layer):
    def __init__(self, ratio=8, **kwargs):
        super(ChannelAttention, self).__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        self.shared_layer_one = tf.keras.layers.Dense(input_shape[-1] // self.ratio,
                                                      activation='relu',
                                                      kernel_initializer='he_normal',
                                                      use_bias=True,
                                                      bias_initializer='zeros')
        self.shared_layer_two = tf.keras.layers.Dense(input_shape[-1],
                                                      kernel_initializer='he_normal',
                                                      use_bias=True,
                                                      bias_initializer='zeros')

    def call(self, inputs):
        avg_pool = tf.reduce_mean(inputs, axis=[1, 2], keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=[1, 2], keepdims=True)

        avg_pool = self.shared_layer_one(avg_pool)
        avg_pool = self.shared_layer_two(avg_pool)

        max_pool = self.shared_layer_one(max_pool)
        max_pool = self.shared_layer_two(max_pool)

        cbam_feature = avg_pool + max_pool
        cbam_feature = tf.keras.activations.sigmoid(cbam_feature)

        return Multiply()([inputs, cbam_feature])

class SpatialAttention(Layer):
    def __init__(self, kernel_size=5, **kwargs):
        super(SpatialAttention, self).__init__(**kwargs)
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.conv2d = Conv2D(filters=1,
                             kernel_size=self.kernel_size,
                             strides=1,
                             padding='same',
                             activation='sigmoid',
                             kernel_initializer='he_normal',
                             use_bias=False)

    def call(self, inputs):
        avg_pool = tf.reduce_mean(inputs, axis=[3], keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=[3], keepdims=True)
        concat = tf.concat([avg_pool, max_pool], axis=-1)
        cbam_feature = self.conv2d(concat)
        return Multiply()([inputs, cbam_feature])

class ParallelAttention(Layer):
    def __init__(self, **kwargs):
        super(ParallelAttention, self).__init__(**kwargs)
        self.channel_attention = ChannelAttention()
        self.spatial_attention = SpatialAttention()

    def call(self, inputs):
        channel_att = self.channel_attention(inputs)
        spatial_att = self.spatial_attention(inputs)
        return Add()([channel_att, spatial_att]) 


# Load the ResNet50 model without the top (fully connected) layers
base_model = EfficientNetV2S(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add channel attention and spatial attention to the base model
combined_att = ParallelAttention()(base_model.output)
output = GlobalAveragePooling2D()(combined_att)
output = Dense(30, activation='softmax')(output)

# Create the model
model = Model(inputs=base_model.input, outputs=output)

# Unfreeze all 15 Layers
for layer in base_model.layers[:-15]:
    layer.trainable = True

# Define the optimizer
optimizer = Adam(learning_rate=0.0001)

# Compile the model with a custom learning rate
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])


model.summary()

model_name = "/content/drive/MyDrive/Research/model/onebranch_canvaswithhand_effiwithGAP.h5"
checkpoint = ModelCheckpoint(model_name,
                            monitor="val_accuracy",
                            mode="max",
                            save_best_only=True,
                            verbose=1)

earlystopping = EarlyStopping(monitor='val_accuracy',
                              min_delta=0,
                              patience=7,
                              verbose=1,
                              restore_best_weights=True)

learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy',
                                            patience=4,
                                            verbose=1,
                                            factor=0.2,
                                            min_lr=0.00000001)
                                            
                                            history = model.fit(train_generator,
                    epochs=50,
                    validation_data=validation_generator,
                    callbacks=[checkpoint,earlystopping,learning_rate_reduction])
 
###################################################################
# Use voting to determine the final prediction
ensemble_prediction = []
for i in range(len(validation_generator.filenames)):  # Iterate over each sample
    # Get the predictions for the i-th sample from all models
    sample_predictions = predictions[:, i, :]
    # Calculate the average prediction probabilities for each class
    avg_prediction = np.mean(sample_predictions, axis=0)
    # Get the index of the class with the highest average probability
    predicted_class = np.argmax(avg_prediction)
    # Append the predicted class to the ensemble predictions
    ensemble_prediction.append(predicted_class)

# Convert the ensemble predictions list into a numpy array
ensemble_prediction = np.array(ensemble_prediction)

# Get true labels
true_labels = validation_generator.classes
accuracy = np.mean(ensemble_prediction == true_labels)
print("Ensemble Accuracy based on Voting:", accuracy) 
