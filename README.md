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
   
   First we need to split our data into three parts train, val and test. why? we will teach our model on train data then to tune the model will test it on validation set and when if it performs well on val set and we test it on test set which is vital because it has never seen that data.  
   
   Now we need our data to be passed in Convolutional Neural Network. remember here we are dealing with two problems one is **Binary Classification which will predict if there is a face in the image** ( not face - 0 , face - 1) and second is **regression problem which will predict the co-ordinates of the bounding boxes**. so we will use *VGG16* which is laready been trained on tons of data so we can use knowledge of that problem ( also called Transfer Learning )and tweak a little bit and add layers one for classification and regression.
   
   Also we need only 2 coordinates to draw or predict a box *topleft or topright and bottom left or bottom right.*
   
   ### What is VGG16 Architecture ?
   >from tensorflow.keras.models import Model
   >
   >from tensorflow.keras.layers import Input, Conv2D, Dense, GlobalMaxPooling2D
   >
   > from tensorflow.keras.applications import VGG16
   
   It is pre-trained model and it contains layers like `Convolution, max pooling, fully connected` and it has Total params: 14,714,688 which has kwnoeldge we can use for our task. when i am saying knowlege i mean weight, filters, layers etc. but we need to tweak it a little bit what we need to is removing the end layer of the model and adding 2 of our layers classification and regression. to do that we have to do something like this :
   
   ```
   vgg = VGG16(include_top = False)
   ```
   
   ![Alt text](https://github.com/rishabh422tiwari/FaceDetection-CNN/blob/main/images/vgg16.png)
   
   ### Define lossed for our classificatipn regression :
   
   In Image classification `Binary cross entropy` loss is pretty common so we will use that.
   
     classloss = tf.keras.losses.BinaryCrossentropy()
   
   For regression we create our own loss which will be based on this formula  which is called `localization loss`:
   
  ![Alt text](https://github.com/rishabh422tiwari/FaceDetection-CNN/blob/main/images/regression%20loss%20function.png)
  
  only difference is we not squaring the whole.
  
  ## Train the model, apply gradient descent and reduce the loss
   
   we break our data into batches of 8 images that means it will take every batch for one time for each epoch and try to reduce loss and make our predictions better. i have set epoch number to 40 i.e for each epoch it will go over every batch 40 times. 
   
   Plotting the losses classification and regression during the training :
   
   ![Alt text](https://github.com/rishabh422tiwari/FaceDetection-CNN/blob/main/images/loss%20plot.png)
   
   See we have little plunging and rise in the loss but at last we were able to reduce the loss pretty much.
  
### Now Make the Prediction 
   
   The out we get for any image will be of two dict class : [0,1] and regression : [x1,y1,x2,y2].
   
   Prediction would look something like this :
   
   ![Alt text](https://github.com/rishabh422tiwari/FaceDetection-CNN/blob/main/images/prediction.png)
   
   ***Tip : Always take input and Visualize output then tweak or fine tune the intermediate layers***
   
   
