At first we selected those data only which were present in training and testing set.
Then,we reduced the size of the images to 120,90 pixels and saved it in npy format.
We designed our model using residual blocks and skip connections to extract deep features from the images.
Then we applied  our model and tuned different hyperparameters like regularizer,optimizers,learning rate decay to improve accuracy.

Dependencies Used-:
1.Numpy
2.Pandas
3.OpenCV
4.Matplotlib
5.OS
6.Keras