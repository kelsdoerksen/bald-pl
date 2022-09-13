import numpy as np
import torch
from torchvision import datasets

class Data:
    def __init__(self, X_train, Y_train, X_test, Y_test, X_pool, Y_pool, handler):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.X_pool = X_pool
        self.Y_pool = Y_pool
        self.handler = handler

        self.n_train = len(X_train)
        self.n_pool = len(X_pool)
        self.n_test = len(X_test)
        
        self.labeled_idxs = np.zeros(self.n_train, dtype=bool)
        self.labeled_pool_idxs = np.zeros(self.n_pool, dtype=bool)
        
    def initialize_labels(self, num):
        # generate initial labeled data based on num of initial labeled samples from X_train we decided
        tmp_idxs = np.arange(self.n_train)
        # shuffle idxs
        np.random.shuffle(tmp_idxs)
        # label a subset
        self.labeled_idxs[tmp_idxs[:num]] = True

    def get_labeled_pool_data(self):
        return np.arange(self.n_pool)[self.labeled_pool_idxs]
    
    def get_labeled_data(self):
        # grab labeled train idxs
        labeled_train_idxs = np.arange(self.n_train)[self.labeled_idxs]
        X_train = self.X_train[labeled_train_idxs]
        Y_train = self.Y_train[labeled_train_idxs]

        # grab labeled pool idxs if any
        labeled_pool_idxs = self.get_labeled_pool_data()
        X_pool = self.X_pool[labeled_pool_idxs]
        Y_pool = self.Y_pool[labeled_pool_idxs]

        # Combine training and labelled pool set
        X_train_combined = torch.cat((X_train, X_pool), 0)
        Y_train_combined = torch.cat((Y_train, Y_pool), 0)
        Y_train_combined = Y_train_combined.long()

        # train model on labeled data
        return self.handler(X_train_combined, Y_train_combined)
    
    def get_unlabeled_data(self):
        # Get unlabeled data from X_pool
        unlabeled_idxs = np.arange(self.n_pool)[~self.labeled_pool_idxs]
        return unlabeled_idxs, self.handler(self.X_pool[unlabeled_idxs], self.Y_pool[unlabeled_idxs])

    def get_train_data(self):
        return self.handler(self.X_train, self.Y_train)
        
    def get_test_data(self):
        return self.handler(self.X_test, self.Y_test)
    
    def cal_test_acc(self, preds):
        return 1.0 * (self.Y_test==preds).sum().item() / self.n_test

    def cal_test_acc_per_class(self, preds):
        labels = self.Y_test
        preds = preds
        acc_per_class = {'0':0, '1':0, '2':0, '3':0, '4':0, '5':0,
                         '6': 0, '7': 0, '8': 0, '9': 0}

        for i in range (10):
            acc = 0
            total_class_samples = len(labels[labels == i])
            for j in range(len(labels)):
                if labels[j] == i:
                    if preds[j] == i:
                        acc += 1 / total_class_samples
            acc_per_class['{}'.format(i)] = acc
        return acc_per_class


def train_split(train_data):
    '''
    Get training data made up of 0-3 digits
    '''
    # split train set to contain only 0-3 digits
    x_train = train_data.data[:40000]
    y_train = train_data.targets[:40000]

    x_zero = x_train[y_train == 0]
    x_one = x_train[y_train == 1]
    x_two = x_train[y_train == 2]
    x_three = x_train[y_train == 3]

    y_zero = torch.from_numpy(np.array([0] * len(x_zero)))
    y_one = torch.from_numpy(np.array([1] * len(x_one)))
    y_two = torch.from_numpy(np.array([2] * len(x_two)))
    y_three = torch.from_numpy(np.array([3] * len(x_three)))

    x_train_03 = torch.cat((x_zero, x_one, x_two, x_three), 0)
    y_train_03 = torch.cat((y_zero, y_one, y_two, y_three), 0)

    return x_train_03, y_train_03

def get_pool_set(train_data):
    '''
    Get pool set made up of 4-9 training data digits
    '''
    # split train set to contain only 4-9 digits
    x_train = train_data.data[:40000]
    y_train = train_data.targets[:40000]

    x_four = x_train[y_train == 4]
    x_five = x_train[y_train == 5]
    x_six = x_train[y_train == 6]
    x_seven = x_train[y_train == 7]
    x_eight = x_train[y_train == 8]
    x_nine = x_train[y_train == 9]

    labels = np.array([])
    for i in range(4,10):
        labels = np.append(labels,np.array([i] * len(x_train[y_train == i])))

    y_train_49 = torch.from_numpy(labels)
    x_train_49 = torch.cat((x_four, x_five, x_six, x_seven,
                            x_eight, x_nine), 0)

    return x_train_49, y_train_49


def get_MNIST(handler):
    raw_train = datasets.MNIST('./data/MNIST', train=True, download=True)
    raw_test = datasets.MNIST('./data/MNIST', train=False, download=True)

    # grab training set of digits 0-3
    x_train, y_train = train_split(raw_train)
    # grab pool set of digits 4-9
    x_pool, y_pool = get_pool_set(raw_train)

    return Data(x_train, y_train, raw_test.data[:40000], raw_test.targets[:40000], x_pool, y_pool, handler)