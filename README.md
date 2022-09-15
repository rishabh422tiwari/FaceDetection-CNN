# Deep Face Detection

A model created to detection face in a image or video. trained on custom dataset curated with the help of **LabelMe, Augmentation and cv2**. This is a Computer vision problem so we need Convolutonal Neural Network to work with images and architecture we are going to use is widely used for image classification **VGG16**.  In order to create data pipeline and the training of the model i have used **tensorflow** sequention API.

## 1. Image Collection
      
   ### First take a picture for our dataset with the help of **CV2** package
   > pip install opencv-python
   > 
   > import cv2

   To take make our dataset little more diverse i have changed background and t-shirts otherwise it will be very easy task for our model to predict the face.
   ![This is an image](https://github.com/rishabh422tiwari/FaceDetection-CNN/blob/main/images/labelme%20image.png)
   
   ### Giving our images label and coordinates of bounding boxes 
   > pip install labelme
   
   It will create label for each of our image and coordinates which contains face in the image. so it will be of dictionary type and contain like the following label : [0,1], coordinate : [45,67 ,432, 523]. *saving it under a JSON file*
   
   ### Now Perform Data Augmentation
   > pip install -U albumentations
   > 
   > import albumentations as alb
   
   Initially i had captured 150 images but look often Computer vision problem or deep learning model need plenty of data to be trained on and give us good results so that is what we are doing augmentating images. Augmenting images means taking **Random Crops, Horizontal Flips, Random brightness contrast, gamma etc** of images that way we don't need to spend our resource to collect data. my images which was *initially 150 now after augmentation is now almost 5000.* 
   
   This is sample images after applying augmentation:
   
   ![This is an image](https://github.com/rishabh422tiwari/FaceDetection-CNN/blob/main/images/Augmented%20image.png)
      

    
## 2. Training deep learning model
   
   Now we need our data to be passed in Convolutional Neural Network. remember here we are dealing with two problems one is **Binary Classification which will predict if there is a face in the image** ( not face - 0 , face - 1) and second is **regression problem which will predict the co-ordinates of the bounding boxes**. so we will use *VGG16* which is laready been trained on tons of data so we can use knowledge of that problem ( also called Transfer Learning )and tweak a little bit and add layers one for classification and regression.
   
   Also we need only 2 coordinates to draw or predict a box *topleft or topright and bottom left or bottom right.*
   
   ### What is VGG16 Architecture ?
   >from tensorflow.keras.models import Model
   >
   >from tensorflow.keras.layers import Input, Conv2D, Dense, GlobalMaxPooling2D
   >
   > from tensorflow.keras.applications import VGG16


       object detection model is 2 diff model - one is classification (what the object is) and 2nd regression model trying to estimate the coordinates of a bounding box (need 2 coordinates to draw a box topleft or topright and bottom left or bottom right )
       
       losses - classification component - binary cross entropy
              - localization loss - keras function api
              
       vgg16 - pretrained - adding the final 2 layers - 1 for class and 1 for regression
       
       
      5 output - class [0,1], regression [x1,y1,x2,y2]


