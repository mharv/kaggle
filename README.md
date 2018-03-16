# kaggle
kaggle notebooks


dogBreedIdentifier_kaggle:

Had a lot of trouble with this one. Originally I built a CNN from scratch, but that performed quite poorly. 
I decided to use a pre-trained model but then ran into a second problem, most of the models used an image of >200x200 pixels. 
When I went to run the script with the required image size, I didn't have enough RAM! 
I think in the future I will have to train the model in parts as training set was so large.

ML_levelOne & ML_levelTwo:

learning some basics using the Machine learning tutorial provided by kaggle. 

Titanic: 

This project started off using various types of different models as seen in titanic1.py 
then the models were ensembled to improve accuracy.

I then decided to see how a neural network would perform.
after one-hot encoding, imputing any missing variables and feature scaling, the data was ready to be processed by the ANN.
I built the ANN using all 691 variables for the input layer (large as the categorical variables had been one-hot encoded, producing a number of variables for each category.),
then, added two hidden layers that that had a relu activation function (after doing alot of research, RELU seems to give great results and seems to out perform every other activation function in most applications),
then used a sigmoid function on the output layer as there were only two possibilities (survived or not)
The loss function used was binary cross-entropy. cross-entropy is useful for classification problems, and more obviously, we are solving a binary classification problem.

