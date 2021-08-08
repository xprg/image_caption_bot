# image_caption_bot

this image caption model has been train on flickr8k dataset.
Using resnet 50 from keras.applications to process image and extract features from them.
Glove 6b.50d has been used for embedding caption words.
LSTM architecture used for modelling the caption sequence.

currently using a basic flask web app for hosting.

Refer the results folder for some sample predictions.

#IMPORTANT : the version of python and other package need be configured as they are different for various platforms.
