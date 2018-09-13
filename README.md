# Deep Neural Network on mnist made in numpy

At the moment the algorithm is using minibatch gradient descent with batch size of 32 reaching 97.43% accuracy on the test set after 100 epochs of training.  
The model can be trained with sgd, rms-prop, momentum, adam although bias corretion is not implemented but this is not really a problem. 
Feel free to play with the hyperparameters. 

## Requirements
* python3
* python3-pip
* python-venv

to run the script run thew following bash:


```bash
python -m venv venv-npNN
source venv-npNN/bin/activate
pip3 install -r freeze.txt
cd src
python3 deepNN.py
```