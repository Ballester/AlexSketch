import random
from numpy import *
from scipy.misc import imread
from scipy.misc import imresize
from scipy.misc import imshow
import config

class Sketch(object):

    folders = []
    train = []
    dataset = {}
    test = []
    counter = 0
    test_counter = 0
    training_size = 0
    test_size = 0
    n_epochs = 0

    train_x = zeros((1, 227,227,3)).astype(float32)
    train_y = zeros((1, 1000))
    xdim = train_x.shape[1:]
    ydim = train_y.shape[1]

    def __init__(self):
        """Open reference file"""
        with open('utils/one-hot_references.txt') as fid:
            # dump
            fid.readline()
            one_hots = []
            one_hots_values = []
            for i in range(0, 57):
                aux = fid.readline()
                aux = aux.split(':')

                one_hots.append(aux[0])
                one_hots_values.append(int(aux[1]))

            for i in range(0, 57):
                self.dataset[one_hots[i]] = one_hots_values[i]
                self.folders.append(one_hots[i])

            random.shuffle(self.folders)
            self.create_sets()
            # self.shuffle_set()
            # self.shuffle_test()
            # print self.train

    """Now training with half and testing with the other half"""
    def create_sets(self, half=False):
        #random.seed(500) ORIGINAL TESTS WITH PARTIAL AMOUNT
        # random.seed(2368712)
        random.seed()
        random.shuffle(self.folders)
        n_folders = 0
        """Used 0 - 28 to train (60 training 20 test)"""
        """Using 29 - N to test (20 test)"""
        for i in range(0, config.partial_amount):
            for n in range(0, 60):
                self.train.append((self.folders[i], n))

        self.training_size = len(self.train)

        for i in range(38, 57):
            for n in range(60, 80):
                self.test.append((self.folders[i], n))

        self.test_size = len(self.test)


    """
    Outputs the next batch, showing one image of each class per epoch
    Considering that the function shuffle_set was never used
    """
    def next_batch_respecting_classes(self, batch_size):
        x_dummy = (random.random((batch_size,)+ self.xdim)/255.).astype(float32)
        images = x_dummy.copy()
        one_hots = []
        for i in range(0, batch_size):
            folder = self.train[self.counter][0]
            name = self.train[self.counter][1]
            self.counter += 60
            if self.counter >= len(self.train):
                self.n_epochs += 1
                #jumping 0 - 60 - 120 ... 1 - 61 - 121 ...
                self.counter = self.n_epochs

            """Turns gray image into RGB image"""
            gray = imresize((imread('sketch_set/' + folder + '/' + str(name) + '.png')[:,:]).astype(float32), (227, 227, 3))
            image = zeros((227, 227, 3))
            image[:,:,0] = image[:,:,1] = image[:,:,2] = gray

            # print image.shape
            images[i,:,:,:] = image
            # imshow(images[i,:,:,:])
            # print images


            """Finds the index of the one-hot encoding by checking the one-hot reference"""
            one_hot = [0.0] * 1000
            l = self.dataset[folder]
            one_hot[l] = 1.0
            one_hots.append(one_hot)

        return images, one_hots
