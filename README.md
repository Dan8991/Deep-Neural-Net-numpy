# Deep Neural Network on mnist made in numpy

At the moment the algorithm is using minibatch gradient descent with batch size of 32 reaching 96,9% accuracy on the test set after 100 epochs of training.  
Need to implement gradient checking to see if there are some errors when calculating gradients.  

## Requirements
* python3
* python3-pip
* python-venv

to run the script run thew following bash:


```bash
python -m venv venv-npNN
source venv_npNN/bin/activate
pip3 install -r freeze.txt
cd src
python3 deepNN.py
```