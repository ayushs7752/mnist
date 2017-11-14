from __future__ import print_function, division
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import csv
import numpy as np
from sklearn import preprocessing as pre
from PIL import Image 




def get_label_matrix(count, x):
    

    np_vector = np.array([int(x) for i in range(count)])
    np_vector = np.reshape(np_vector, (count, 1))
    return np_vector




def read_digit_x(digit, countTrain, countTest):


    digit_path = '/Users/ayushs/Desktop/hw3/hw2_resources2/data/mnist_digit_', str(digit),'.csv'
    digit_path= "".join(digit_path)


    np_matrix_train = np.zeros((1,784))
    np_matrix_test = np.zeros((1,784))
    # print ("sizes check before -- ", np.shape(np_matrix_test),np.shape(np_matrix_train))

    num = 1



    with open(digit_path, 'rb') as f:
        reader = csv.reader(f)
        for row in reader:
            vector = str(row).split()
            vector[0], vector[-1] = 0,0
            np_vector = np.array([int(i) for i in vector])
            # print (np_vector)
            # print (np_matrix.size())
           

            np_vector = np.reshape(np_vector,(1,784))

            if num <= countTrain:
                np_matrix_train = np.append(np_matrix_train, np_vector, axis = 0)

            elif num > countTrain and num <=countTrain + countTest:
                np_matrix_test = np.append(np_matrix_test, np_vector, axis = 0)
                # print ("shape here- -", np.shape(np_matrix_test))

            else:
                break
           
            num+=1 


    # print ("sizes check before ", np.shape(np_matrix_test),np.shape(np_matrix_train))
    

    np_matrix_train = np_matrix_train[1:,:] 
    np_matrix_test = np_matrix_test[1:,:]           

    label_matrix_train = get_label_matrix(countTrain, digit)
    label_matrix_test = get_label_matrix(countTest, digit)

    # print ("sizes check here", np.shape(np_matrix_test),np.shape(np_matrix_train), np.shape(label_matrix_test), np.shape(label_matrix_train))
    # print ("size train matrix: ", np.shape(np_matrix_train))

    return np_matrix_train, label_matrix_train, np_matrix_test, label_matrix_test






def get_data(trainSize, testSize):

    np_matrix_train = np.zeros((1,784))
    np_matrix_test = np.zeros((1,784))
 
    label_matrix_test = np.zeros((1,1))
    label_matrix_train = np.zeros((1,1))

    for i in range(10):

        data_call = read_digit_x(i, trainSize, testSize)
        
        npTrainAppend = data_call[0]
        # print ("size np train append ", np.shape(npTrainAppend) )
        np_matrix_train = np.concatenate((np_matrix_train, npTrainAppend), axis = 0)
        

        npTestAppend = data_call[2]
        np_matrix_test = np.concatenate((np_matrix_test, npTestAppend), axis = 0)

        label_matrix_train_append = data_call[1]
        label_matrix_train = np.concatenate((label_matrix_train, label_matrix_train_append), axis = 0)

        label_matrix_test_append = data_call[3]
        label_matrix_test = np.concatenate((label_matrix_test, label_matrix_test_append), axis = 0)
    


    # print ("size here- ", np.shape(np_matrix_train))


    # img = Image.fromarray(np_matrix_test[200,:].reshape((28,28)))
    # img.show()
    # print ("label check 1 ", label_matrix_test[40,:])


    np_matrix_train = np_matrix_train[1:,:] 
    np_matrix_test = np_matrix_test[1:,:] 

    label_matrix_train = label_matrix_train[1:,:] 
    label_matrix_test = label_matrix_test[1:,:] 


    # img = Image.fromarray(np_matrix_test[1,:].reshape((28,28)))
    # img.show()
    # print ("label check ", label_matrix_test[1,:])

    # print ("matrix sizes", np.shape(np_matrix_train), np.shape(np_matrix_test))
    # print ("label sizes ", np.shape(label_matrix_train), np.shape(label_matrix_test))




    np_matrix_train = 2*np_matrix_train/255.0 -1 
    np_matrix_test = 2*np_matrix_test/255.0 - 1 

    # print ("np matrix train ", np_matrix_train[0,:])
    # print ("matrix sizes", np.shape(np_matrix_train), np.shape(np_matrix_test))



    return np_matrix_train, label_matrix_train, np_matrix_test, label_matrix_test



def get_train_data(trainSize,testSize):
    data_call = get_data(trainSize, testSize)
    return data_call[0], data_call[1]



def get_test_data(trainSize, testSize):
    data_call = get_data(trainSize, testSize)
    return data_call[2], data_call[3]



class MNISTdataset():
    def __init__(self, trainSize, testSize, transform=None):
        
        self.transform = transform
        self.data = get_train_data(trainSize, testSize)
        print('Loaded Dataset.')


    def __len__(self):
        # print ("len:", np.shape(self.data[0]))
        return len(self.data[1])

    def __getitem__(self, idx):
        # print ("idx:", idx)
        data = self.data[0][idx-1, :]

        # data_float = data.astype(float)
        image = torch.from_numpy(data)
        # print (type(image))
        if self.transform:
            image = self.transform(image)

        image = image.float()


        label = int(self.data[1][idx,:])

        return image, label

class MNISTTestset():
    """
    MiniPlaces dataset for the test set.
    The test set only has images, which are in the images_dir.
    There are no labels -- these are provided by the model.
    """
    def __init__(self, trainSize, testSize, transform=None):
        self.data = get_test_data(trainSize, testSize)
        self.transform = transform
        print('Loaded TestSet')

    def __len__(self):
        return len(self.data[1])

    def __getitem__(self, idx):
        """ Returns a transformed image and filename. """
        data = self.data[0][idx-1, :]
        # data_float = data.astype(float)



        image = torch.from_numpy(data)
        # print (type(image))
        if self.transform:
            image = self.transform(image)
        image = image.float()

        

        label = int(self.data[1][idx,:])

        return image, label










