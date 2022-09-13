import argparse
import numpy as np
import torch
from utils import get_dataset, get_net, get_strategy
from pprint import pprint
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1, help="random seed")
    parser.add_argument('--n_init_labeled', type=int, default=10000, help="number of init labeled samples")
    parser.add_argument('--n_query', type=int, default=1000, help="number of queries per round")
    parser.add_argument('--n_round', type=int, default=10, help="number of rounds")
    parser.add_argument('--dataset_name', type=str, default="MNIST", choices=["MNIST", "FashionMNIST", "SVHN", "CIFAR10"], help="dataset")
    parser.add_argument('--strategy_name', type=str, default="BALDDropout",
                        choices=["RandomSampling",
                                 "LeastConfidence",
                                 "MarginSampling",
                                 "EntropySampling",
                                 "LeastConfidenceDropout",
                                 "MarginSamplingDropout",
                                 "EntropySamplingDropout",
                                 "KMeansSampling",
                                 "KCenterGreedy",
                                 "BALDDropout",
                                 "AdversarialBIM",
                                 "AdversarialDeepFool"], help="query strategy")
    args = parser.parse_args()
    pprint(vars(args))
    print()

    # fix random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.enabled = False

    # device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # load train and test data
    dataset = get_dataset(args.dataset_name)
    # load network
    net = get_net(args.dataset_name, device)
    # load strategy
    strategy = get_strategy(args.strategy_name)(dataset, net)

    # start experiment
    dataset.initialize_labels(args.n_init_labeled)
    print(f"number of labeled pool: {args.n_init_labeled}")
    print(f"number of unlabeled pool: {dataset.n_pool}")
    print(f"number of testing pool: {dataset.n_test}")
    print()

    # round 0 accuracy
    model_acc = []
    print("Round 0")
    strategy.train()
    preds = strategy.predict(dataset.get_test_data())
    print(f"Round 0 testing accuracy: {dataset.cal_test_acc(preds)}")
    # Saving model accuracy to plot
    model_acc.append(dataset.cal_test_acc(preds))
    # calculate accuracy per class
    per_class_acc = dataset.cal_test_acc_per_class(preds)
    print('Accuracy per class 0-9: {}'.format(per_class_acc.values()))

    uncertainties_per_rounds = []
    for rd in range(1, args.n_round+1):
        print(f"Round {rd}")

        # query and return the idxs for the top n samples based on strategy
        query_idxs = strategy.query(args.n_query)

        # update pool labels
        strategy.update(query_idxs)
        # re-train with added pool data
        strategy.train()

        # calculate accuracy on test set
        preds = strategy.predict(dataset.get_test_data())
        print(f"Round {rd} testing accuracy: {dataset.cal_test_acc(preds)}")
        model_acc.append(dataset.cal_test_acc(preds))

        # Calculate entropy as a measure of uncertainty from predictions on test set per round
        data =  dataset.get_test_data()
        probs = strategy.predict_prob_dropout(data)
        uncertainties = strategy.get_test_uncertainties(data, probs)
        uncertainties_per_rounds.append(uncertainties)

        # calculate accuracy per class
        per_class_acc = dataset.cal_test_acc_per_class(preds)
        print('Accuracy per class 0-9: {}'.format(per_class_acc.values()))

    # Let's plot first and last hist for only digit 8 as a test (in the pool set) as an example
    # messy, just to observe/play around with stuff for now
    first = uncertainties_per_rounds[0]
    last = uncertainties_per_rounds[-1]
    df_filt_1 = first.loc[first['label'] == 8]
    df_filt_1['uncertainties'].hist(label='first round', alpha=0.2)
    df_filt_end = last.loc[last['label'] == 8]
    df_filt_end['uncertainties'].hist(label='last round', alpha=0.2)
    plt.xlabel('Entropy')
    plt.ylabel('Number of Samples')
    plt.legend(loc='upper right')
    plt.show()