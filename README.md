# Deep Face Detection

A model created to detection face in a image or video. trained on custom dataset curated with the help of **LabelMe, Augmentation and cv2**. In order to create data pipeline and the training of the model i have used **tensorflow** sequention API.

##1. Image Collection
    using webcam collect imgaes
    annotate images i.e bounding boxes LabelMe
    data augmentation augmentation - random crop, brightness, flip, gamma
    
2. Training deep learning model
       object detection model is 2 diff model - one is classification (what the object is) and 2nd regression model trying to estimate the coordinates of a bounding box (need 2 coordinates to draw a box topleft or topright and bottom left or bottom right )
       
       losses - classification component - binary cross entropy
              - localization loss - keras function api
              
       vgg16 - pretrained - adding the final 2 layers - 1 for class and 1 for regression
       
       
      5 output - class [0,1], regression [x1,y1,x2,y2]


