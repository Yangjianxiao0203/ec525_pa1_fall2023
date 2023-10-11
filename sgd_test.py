import numpy as np
from sklearn.datasets import fetch_openml

import ops_impl as ops
from sgd import SGD
from variable import Variable
import os
# load mnist data:
# def load_mnist():
#     print("loading mnist data....")
#     mnist = fetch_openml('mnist_784', data_home='./data', as_frame=False)
#     mnist.target = np.array([int(t) for t in mnist.target])
#
#     print("done!")
#     return mnist
class MNISTContainer:
    def __init__(self, data, target):
        self.data = data
        self.target = target
def load_mnist():
    print("loading mnist data....")

    data_dir = "./data/"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    data_file = os.path.join(data_dir, 'mnist_784.npz')
    if not os.path.exists(data_file):
        print("Downloading MNIST dataset...")
        mnist = fetch_openml('mnist_784', data_home=data_dir,as_frame=False)
        mnist.target = np.array([int(t) for t in mnist.target])
        print("Done!")
        print("Saving...")
        np.savez(data_file, data=mnist.data, target=mnist.target)
        print("Done!")
    mnist = np.load(data_file)
    print("MNIST dataset loaded.")
    return MNISTContainer(mnist['data'], mnist['target'])


def loss_fn(params, data):
    '''computes hinge for linear classification of MNIST digits.
    
    args:
        params: list containing [weights, bias]
            where weights is a 10x784 Variable and
            bias is a scalar Variable.
        data: list containing [features, label]
            where features is a 784 dimensional numpy array
            and label is an integer
        
    returns:
        loss, correct
            where loss is a Variable representing the hinge loss
            of the 10-dimenaional scores where
            scores[i] = dot(weights[i] , features) + bias
            and correct is a float that is 1.0 if scores[label] is the largest
            score and 0.0 otherwise.
    '''

    ### YOUR CODE HERE ###
    features, label = data # features: 784 * 1, label: scaler
    features = Variable(features)
    scores = get_scores(features, params) # 10 * 1
    hinge = ops.HingeLoss(label)
    loss = hinge(scores) # scaler
    correct = float(np.argmax(scores.data) == label)

    #for test
    # if correct == 1.0 :
    #     print("scores max", np.argmax(scores))
    #     print("label: ", label)
    #     print("loss:",loss.data)

    return loss, correct

def get_scores(features, params):
    weights, bias = params
    return ops.matmul(weights, features) + bias

def get_normal(shape):
    return np.random.normal(np.zeros(shape))

def train_mnist(learning_rate, epochs, mnist):
    print("training linear classifier...")
    running_accuracy = 0.0
    it = 0

    TRAINING_SIZE = 60000
    TESTING_SIZE = 10000

    params = [Variable(np.zeros((10, 784))), Variable(np.zeros((10, 1)))]

    for it in range(epochs * TRAINING_SIZE):
        data = [mnist.data[it % 60000].reshape(-1, 1)/255.0, mnist.target[it % 60000]]
        params, correct = SGD(loss_fn, params, data, learning_rate)
        running_accuracy += (correct - running_accuracy)/(it + 1.0) #TODO: why this formula?
        if (it+1) % 10000 == 0:
            print("iteration: {}, current train accuracy: {}".format(it+1, running_accuracy))

    running_accuracy = 0.0
    print("running evaluation...")
    for it in range(TESTING_SIZE):
        data = [mnist.data[it + 60000].reshape(-1, 1)/255.0, mnist.target[it + 60000]]
        loss, correct = loss_fn(params, data)
        running_accuracy += (correct - running_accuracy)/(it + 1.0)
    print("eval accuracy: ", running_accuracy)


if __name__ == '__main__':
    mnist_data = load_mnist()
    train_mnist(0.01, 2, mnist_data)