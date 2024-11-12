"""
Coauthors: Haoyin Xu
           Yu-Chung Peng
           Madi Kusmanov
           Adway Kanhere
"""
from toolbox_ziyan import *
import argparse
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from skopt import BayesSearchCV
from skopt.callbacks import DeadlineStopper
from FMin import fmin
import xgboost as xgb
from hpbandster.optimizers import BOHB
import hpbandster.core.nameserver as hpns
from hpbandster.core.worker import Worker
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter, UniformFloatHyperparameter, CategoricalHyperparameter
import re

import pandas as pd
import torchvision.models as models
import random
import pickle
import warnings

warnings.filterwarnings("ignore")

np.random.seed(317)

from smac import Scenario
from pathlib import Path
from smac.facade import AbstractFacade
import matplotlib.pyplot as plt
from smac import MultiFidelityFacade as MFFacade
from smac.intensifier.hyperband import Hyperband
from smac.intensifier.successive_halving import SuccessiveHalving


def run_GBT():
    gbt_kappa = []
    gbt_ece = []
    gbt_train_time = []
    gbt_test_time = []
    gbt_probs_labels = []
    storage_dict = {}

    for classes in classes_space:
        d1 = {}
        for samples in samples_space:
            l3 = []
            # initial GBT model
            gbt_model = xgb.XGBClassifier(
                n_estimators=600,
                max_depth=32,
                learning_rate=0.1,
                objective='multi:softprob',
                random_state=317
            )

            cohen_kappa, ece, train_time, test_time, test_probs, test_labels, test_preds = run_gbt_image_set(
                gbt_model,
                fsdk18_train_images,
                fsdk18_train_labels,
                fsdk18_test_images,
                fsdk18_test_labels,
                samples,
                classes,
            )

            # Store the results
            gbt_kappa.append(cohen_kappa)
            gbt_ece.append(ece)
            gbt_train_time.append(train_time)
            gbt_test_time.append(test_time)
            classes = sorted(classes)
            gbt_probs_labels.append("Classes:" + str(classes))
            gbt_probs_labels.append("Sample size:" + str(samples))
            for i in range(len(test_probs)):
                gbt_probs_labels.append("Posteriors:"+str(test_probs[i]) + ", " + "Test Labels:" + str(test_labels[i]))
            gbt_probs_labels.append(" \n")
            for i in range(len(test_probs)):
                l3.append([test_probs[i].tolist(), test_labels[i]])
            d1[samples] = l3
        storage_dict[tuple(sorted(classes))] = d1

    switched_storage_dict = {}
    for classes, class_data in storage_dict.items():
        for samples, data in class_data.items():
            if samples not in switched_storage_dict:
                switched_storage_dict[samples] = {}
            if classes not in switched_storage_dict[samples]:
                switched_storage_dict[samples][classes] = data
    with open(prefix +'gbt_switched_storage_dict.pkl', 'wb') as f:
        pickle.dump(switched_storage_dict, f)
    # save the model
    with open(prefix + 'gbt_org.pkl', 'wb') as f:
        pickle.dump(gbt_model, f)
    print("gbt finished")
    write_result(prefix + "gbt_kappa_best.txt", gbt_kappa)
    write_result(prefix + "gbt_ece.txt", gbt_ece)
    write_result(prefix + "gbt_train_time.txt", gbt_train_time)
    write_result(prefix + "gbt_test_time.txt", gbt_test_time)
    write_result(prefix + "gbt_probs&labels.txt", gbt_probs_labels)
    write_json(prefix + "gbt_kappa_best.json", gbt_kappa)
    write_json(prefix + "gbt_ece.json", gbt_ece)
    write_json(prefix + "gbt_train_time.json", gbt_train_time)
    write_json(prefix + "gbt_test_time.json", gbt_test_time)


def run_naive_rf():
    naive_rf_kappa = []
    naive_rf_ece = []
    naive_rf_train_time = []
    naive_rf_test_time = []
    navie_rf_probs_labels = []
    storage_dict = {}

    for classes in classes_space:
        d1 = {}
        # cohen_kappa vs num training samples (naive_rf)
        for samples in samples_space:
            l3 = []            
            # train data
            RF_best = RandomForestClassifier(n_estimators=400, max_depth=16, min_samples_split=2, min_samples_leaf=1, max_features=None ,n_jobs=-1, random_state=317)

            cohen_kappa, ece, train_time, test_time, test_probs, test_labels, test_preds = run_rf_image_set(
                RF_best,
                fsdk18_train_images,
                fsdk18_train_labels,
                fsdk18_test_images,
                fsdk18_test_labels,
                samples,
                classes,
            )

            # Store the results
            naive_rf_kappa.append(cohen_kappa)
            naive_rf_ece.append(ece)
            naive_rf_train_time.append(train_time)
            naive_rf_test_time.append(test_time)

            classes = sorted(classes)
            navie_rf_probs_labels.append("Classes:" + str(classes))

            navie_rf_probs_labels.append("Sample size:" + str(samples))

            for i in range(len(test_probs)):
                navie_rf_probs_labels.append("Posteriors:"+str(test_probs[i]) + ", " + "Test Labels:" + str(test_labels[i]))
            navie_rf_probs_labels.append(" \n")

            for i in range(len(test_probs)):
                l3.append([test_probs[i].tolist(), test_labels[i]])

            d1[samples] = l3

        storage_dict[tuple(sorted(classes))] = d1

    # switch the classes and sample sizes
    switched_storage_dict = {}

    for classes, class_data in storage_dict.items():

        for samples, data in class_data.items():

            if samples not in switched_storage_dict:
                switched_storage_dict[samples] = {}

            if classes not in switched_storage_dict[samples]:
                switched_storage_dict[samples][classes] = data

    with open(prefix +'rf_switched_storage_dict.pkl', 'wb') as f:
        pickle.dump(switched_storage_dict, f)

    # save the model
    with open(prefix + 'naive_rf_org.pkl', 'wb') as f:
        pickle.dump(RF_best, f)

    print("naive_rf finished")
    write_result(prefix + "naive_rf_kappa_best.txt", naive_rf_kappa)
    write_result(prefix + "naive_rf_ece.txt", naive_rf_ece)
    write_result(prefix + "naive_rf_train_time.txt", naive_rf_train_time)
    write_result(prefix + "naive_rf_test_time.txt", naive_rf_test_time)
    write_result(prefix + "naive_rf_probs&labels.txt", navie_rf_probs_labels)
    write_json(prefix + "naive_rf_kappa_best.json", naive_rf_kappa)
    write_json(prefix + "naive_rf_ece.json", naive_rf_ece)
    write_json(prefix + "naive_rf_train_time.json", naive_rf_train_time)
    write_json(prefix + "naive_rf_test_time.json", naive_rf_test_time)


def run_cnn32():
    cnn32_kappa = []
    cnn32_ece = []
    cnn32_train_time = []
    cnn32_test_time = []
    cnn32_probs_labels = []
    storage_dict = {}

    for classes in classes_space:
        d1 = {}

        # cohen_kappa vs num training samples (cnn32)
        for samples in samples_space:
            l3 = []
            # train data
            cnn32 = SimpleCNN32Filter(len(classes))
            # 3000 samples, 80% train is 2400 samples, 20% test
            train_images = trainx.copy()
            train_labels = trainy.copy()
            # reshape in 4d array
            test_images = testx.copy()
            test_labels = testy.copy()

            (
                train_images,
                train_labels,
                valid_images,
                valid_labels,
                test_images,
                test_labels,
            ) = prepare_data(
                train_images, train_labels, test_images, test_labels, samples, classes
            )

            cohen_kappa, ece, train_time, test_time, test_probs, test_labels, test_preds = run_dn_image_es(
                cnn32,
                train_images,
                train_labels,
                valid_images,
                valid_labels,
                test_images,
                test_labels,
                optimizer_name="adam",
                epochs=100,
                batch=1024,
                lr=0.001,
            )

            # Store the results
            cnn32_kappa.append(cohen_kappa)
            cnn32_ece.append(ece)
            cnn32_train_time.append(train_time)
            cnn32_test_time.append(test_time)

            actual_test_labels = []
            for i in range(len(test_labels)):
                actual_test_labels.append(int(classes[test_labels[i]]))

            sorted_classes = sorted(classes)
            cnn32_probs_labels.append("Classes:" + str(sorted_classes))

            cnn32_probs_labels.append("Sample size:" + str(samples))
            
            actual_preds = []
            for i in range(len(test_preds)):
                actual_preds.append(int(sorted_classes[test_preds[i].astype(int)]))

            for i in range(len(test_probs)):
                cnn32_probs_labels.append("Posteriors:"+str(test_probs[i]) + ", " + "Test Labels:" + str(actual_test_labels[i]))
            cnn32_probs_labels.append(" \n")

            for i in range(len(test_probs)):
                l3.append([test_probs[i].tolist(), actual_test_labels[i]])

            d1[samples] = l3

        storage_dict[tuple(sorted(classes))] = d1

    # switch the classes and sample sizes
    switched_storage_dict = {}
    
    for classes, class_data in storage_dict.items():
        for samples, data in class_data.items():

            if samples not in switched_storage_dict:
                switched_storage_dict[samples] = {}

            if classes not in switched_storage_dict[samples]:
                switched_storage_dict[samples][classes] = data

    with open(prefix +'cnn32_switched_storage_dict.pkl', 'wb') as f:
        pickle.dump(switched_storage_dict, f)

    # save the model
    with open(prefix + 'cnn32.pkl', 'wb') as f:
        pickle.dump(cnn32, f)

    print("cnn32 finished")
    write_result(prefix + "cnn32_kappa_best.txt", cnn32_kappa)
    write_result(prefix + "cnn32_ece.txt", cnn32_ece)
    write_result(prefix + "cnn32_train_time.txt", cnn32_train_time)
    write_result(prefix + "cnn32_test_time.txt", cnn32_test_time)
    write_result(prefix + "cnn32_probs&labels.txt", cnn32_probs_labels)
    write_json(prefix + "cnn32_kappa_best.json", cnn32_kappa)
    write_json(prefix + "cnn32_ece.json", cnn32_ece)
    write_json(prefix + "cnn32_train_time.json", cnn32_train_time)
    write_json(prefix + "cnn32_test_time.json", cnn32_test_time)


def run_cnn32_2l():
    cnn32_2l_kappa = []
    cnn32_2l_ece = []
    cnn32_2l_train_time = []
    cnn32_2l_test_time = []
    cnn32_2l_probs_labels = []
    storage_dict = {}

    for classes in classes_space:
        d1 = {}

        # cohen_kappa vs num training samples (cnn32_2l)
        for samples in samples_space:
            l3 = []
            # train data
            cnn32_2l = SimpleCNN32Filter2Layers(len(classes))
            # 3000 samples, 80% train is 2400 samples, 20% test
            train_images = trainx.copy()
            train_labels = trainy.copy()
            # reshape in 4d array
            test_images = testx.copy()
            test_labels = testy.copy()

            (
                train_images,
                train_labels,
                valid_images,
                valid_labels,
                test_images,
                test_labels,
            ) = prepare_data(
                train_images, train_labels, test_images, test_labels, samples, classes
            )

            cohen_kappa, ece, train_time, test_time, test_probs, test_labels, test_preds = run_dn_image_es(
                cnn32_2l,
                train_images,
                train_labels,
                valid_images,
                valid_labels,
                test_images,
                test_labels,
                optimizer_name="adam",
                batch=256,
                epochs=100,
                lr=0.01,
            )

            # Store the results
            cnn32_2l_kappa.append(cohen_kappa)
            cnn32_2l_ece.append(ece)
            cnn32_2l_train_time.append(train_time)
            cnn32_2l_test_time.append(test_time)

            actual_test_labels = []
            for i in range(len(test_labels)):
                actual_test_labels.append(int(classes[test_labels[i]]))

            sorted_classes = sorted(classes)
            cnn32_2l_probs_labels.append("Classes:" + str(classes))

            cnn32_2l_probs_labels.append("Sample size:" + str(samples))

            actual_preds = []
            for i in range(len(test_preds)):
                actual_preds.append(int(sorted_classes[test_preds[i].astype(int)]))
            
            for i in range(len(test_probs)):
                cnn32_2l_probs_labels.append("Posteriors:"+str(test_probs[i]) + ", " + "Test Labels:" + str(actual_test_labels[i]))
            cnn32_2l_probs_labels.append(" \n")

            for i in range(len(test_probs)):
                l3.append([test_probs[i].tolist(), actual_test_labels[i]])

            d1[samples] = l3

        storage_dict[tuple(sorted(classes))] = d1

    # switch the classes and sample sizes
    switched_storage_dict = {}

    for classes, class_data in storage_dict.items():
        for samples, data in class_data.items():

            if samples not in switched_storage_dict:
                switched_storage_dict[samples] = {}

            if classes not in switched_storage_dict[samples]:
                switched_storage_dict[samples][classes] = data

    with open(prefix + 'cnn32_2l_switched_storage_dict.pkl', 'wb') as f:
        pickle.dump(switched_storage_dict, f)

    # save the model
    with open(prefix + 'cnn32_2l.pkl', 'wb') as f:
        pickle.dump(cnn32_2l, f)

    print("cnn32_2l finished")
    write_result(prefix + "cnn32_2l_kappa_best.txt", cnn32_2l_kappa)
    write_result(prefix + "cnn32_2l_ece.txt", cnn32_2l_ece)
    write_result(prefix + "cnn32_2l_train_time.txt", cnn32_2l_train_time)
    write_result(prefix + "cnn32_2l_test_time.txt", cnn32_2l_test_time)
    write_result(prefix + "cnn32_2l_probs&labels.txt", cnn32_2l_probs_labels)
    write_json(prefix + "cnn32_2l_kappa_best.json", cnn32_2l_kappa)
    write_json(prefix + "cnn32_2l_ece.json", cnn32_2l_ece)
    write_json(prefix + "cnn32_2l_train_time.json", cnn32_2l_train_time)
    write_json(prefix + "cnn32_2l_test_time.json", cnn32_2l_test_time)


def run_cnn32_5l():
    cnn32_5l_kappa = []
    cnn32_5l_ece = []
    cnn32_5l_train_time = []
    cnn32_5l_test_time = []
    cnn32_5l_probs_labels = []
    storage_dict = {}

    for classes in classes_space:
        d1 = {}

        # cohen_kappa vs num training samples (cnn32_5l)
        for samples in samples_space:
            l3 = []
            # train data
            cnn32_5l = SimpleCNN32Filter5Layers(len(classes))
            # 3000 samples, 80% train is 2400 samples, 20% test
            train_images = trainx.copy()
            train_labels = trainy.copy()
            # reshape in 4d array
            test_images = testx.copy()
            test_labels = testy.copy()

            (
                train_images,
                train_labels,
                valid_images,
                valid_labels,
                test_images,
                test_labels,
            ) = prepare_data(
                train_images, train_labels, test_images, test_labels, samples, classes
            )

            cohen_kappa, ece, train_time, test_time, test_probs, test_labels, test_preds = run_dn_image_es(
                cnn32_5l,
                train_images,
                train_labels,
                valid_images,
                valid_labels,
                test_images,
                test_labels,
                optimizer_name="sgd",
                lr=0.001,
                epochs=80,
                batch=32,
            )

            # Store the results
            cnn32_5l_kappa.append(cohen_kappa)
            cnn32_5l_ece.append(ece)
            cnn32_5l_train_time.append(train_time)
            cnn32_5l_test_time.append(test_time)

            actual_test_labels = []
            for i in range(len(test_labels)):
                actual_test_labels.append(int(classes[test_labels[i]]))

            sorted_classes = sorted(classes)
            cnn32_5l_probs_labels.append("Classes:" + str(classes))

            cnn32_5l_probs_labels.append("Sample size:" + str(samples))

            actual_preds = []
            for i in range(len(test_preds)):
                actual_preds.append(int(sorted_classes[test_preds[i].astype(int)]))

            for i in range(len(test_probs)):
                cnn32_5l_probs_labels.append("Posteriors:"+str(test_probs[i]) + ", " + "Test Labels:" + str(actual_test_labels[i]))
            cnn32_5l_probs_labels.append(" \n")

            for i in range(len(test_probs)):
                l3.append([test_probs[i].tolist(), actual_test_labels[i]])

            d1[samples] = l3

        storage_dict[tuple(sorted(classes))] = d1

    # switch the classes and sample sizes
    switched_storage_dict = {}

    for classes, class_data in storage_dict.items():
        for samples, data in class_data.items():

            if samples not in switched_storage_dict:
                switched_storage_dict[samples] = {}

            if classes not in switched_storage_dict[samples]:
                switched_storage_dict[samples][classes] = data

    with open(prefix + 'cnn32_5l_switched_storage_dict.pkl', 'wb') as f:
        pickle.dump(switched_storage_dict, f)

    # save the model
    with open(prefix + 'cnn32_5l.pkl', 'wb') as f:
        pickle.dump(cnn32_5l, f)

    print("cnn32_5l finished")
    write_result(prefix + "cnn32_5l_kappa_best.txt", cnn32_5l_kappa)
    write_result(prefix + "cnn32_5l_ece.txt", cnn32_5l_ece)
    write_result(prefix + "cnn32_5l_train_time.txt", cnn32_5l_train_time)
    write_result(prefix + "cnn32_5l_test_time.txt", cnn32_5l_test_time)
    write_result(prefix + "cnn32_5l_probs&labels.txt", cnn32_5l_probs_labels)
    write_json(prefix + "cnn32_5l_kappa.json", cnn32_5l_kappa)
    write_json(prefix + "cnn32_5l_ece.json", cnn32_5l_ece)
    write_json(prefix + "cnn32_5l_train_time.json", cnn32_5l_train_time)
    write_json(prefix + "cnn32_5l_test_time.json", cnn32_5l_test_time)
    
def run_resnet18():
    resnet18_kappa = []
    resnet18_ece = []
    resnet18_train_time = []
    resnet18_test_time = []
    resnet18_probs_labels = []
    storage_dict = {}

    for classes in classes_space:
        d1 = {}

        # cohen_kappa vs num training samples (resnet18)
        for samples in samples_space:
            l3 = []
            resnet = models.resnet18(pretrained=True)

            num_ftrs = resnet.fc.in_features
            resnet.fc = nn.Linear(num_ftrs, len(classes))
            # train data
            # 3000 samples, 80% train is 2400 samples, 20% test
            train_images = trainx.copy()
            train_labels = trainy.copy()
            # reshape in 4d array
            test_images = testx.copy()
            test_labels = testy.copy()

            (
                train_images,
                train_labels,
                valid_images,
                valid_labels,
                test_images,
                test_labels,
            ) = prepare_data(
                train_images, train_labels, test_images, test_labels, samples, classes
            )

            # need to duplicate channel because batch norm cant have 1 channel images
            train_images = torch.cat((train_images, train_images, train_images), dim=1)
            valid_images = torch.cat((valid_images, valid_images, valid_images), dim=1)
            test_images = torch.cat((test_images, test_images, test_images), dim=1)

            cohen_kappa, ece, train_time, test_time, test_probs, test_labels, test_preds = run_dn_image_es(
                resnet,
                train_images,
                train_labels,
                valid_images,
                valid_labels,
                test_images,
                test_labels,
                epochs=100,
                lr=0.001,
                batch=32,
                optimizer_name="sgd",
            )

            # Store the results
            resnet18_kappa.append(cohen_kappa)
            resnet18_ece.append(ece)
            resnet18_train_time.append(train_time)
            resnet18_test_time.append(test_time)

            actual_test_labels = []
            for i in range(len(test_labels)):
                actual_test_labels.append(int(classes[test_labels[i]]))

            sorted_classes = sorted(classes)
            resnet18_probs_labels.append("Classes:" + str(classes))

            resnet18_probs_labels.append("Sample size:" + str(samples))

            actual_preds = []
            for i in range(len(test_preds)):
                actual_preds.append(int(sorted_classes[test_preds[i].astype(int)]))

            for i in range(len(test_probs)):
                resnet18_probs_labels.append("Posteriors:"+str(test_probs[i]) + ", " + "Test Labels:" + str(actual_test_labels[i]))
            resnet18_probs_labels.append(" \n")

            for i in range(len(test_probs)):
                l3.append([test_probs[i].tolist(), actual_test_labels[i]])
        
            d1[samples] = l3
        storage_dict[tuple(sorted(classes))] = d1

    # switch the classes and sample sizes
    switched_storage_dict = {}

    for classes, class_data in storage_dict.items():
        for samples, data in class_data.items():

            if samples not in switched_storage_dict:
                switched_storage_dict[samples] = {}

            if classes not in switched_storage_dict[samples]:
                switched_storage_dict[samples][classes] = data

    with open(prefix + 'resnet18_switched_storage_dict.pkl', 'wb') as f:
        pickle.dump(switched_storage_dict, f)

    # save the model
    with open(prefix + 'resnet18.pkl', 'wb') as f:
        pickle.dump(resnet, f)


    print("resnet18 finished")
    write_result(prefix + "resnet18_kappa_best.txt", resnet18_kappa)
    write_result(prefix + "resnet18_ece.txt", resnet18_ece)
    write_result(prefix + "resnet18_train_time.txt", resnet18_train_time)
    write_result(prefix + "resnet18_test_time.txt", resnet18_test_time)
    write_result(prefix + "resnet18_probs&labels.txt", resnet18_probs_labels)
    write_json(prefix + "resnet18_kappa.json", resnet18_kappa)
    write_json(prefix + "resnet18_ece.json", resnet18_ece)
    write_json(prefix + "resnet18_train_time.json", resnet18_train_time)
    write_json(prefix + "resnet18_test_time.json", resnet18_test_time)


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", help="class number")
    parser.add_argument("-f", help="feature type")
    parser.add_argument("-data", help="audio files location")
    parser.add_argument("-labels", help="labels file location")
    args = parser.parse_args()
    n_classes = int(args.m)
    feature_type = str(args.f)

    train_folder = str(args.data)
    train_label = pd.read_csv(str(args.labels))

    # select subset of data that only contains 300 samples per class
    labels_chosen = train_label[
        train_label["label"].map(train_label["label"].value_counts() >= 0)
    ]

    # # get the sample size of each class
    # sample_size = labels_chosen.label.value_counts().to_dict()
    # print("sample_sizes:", sample_size)
    # print("labels_chosen:", labels_chosen.iloc[:, 1:2].value_counts())

    training_files = []
    for file in os.listdir(train_folder):
        for x in labels_chosen.fname.to_list():
            if file.endswith(x):
                training_files.append(file)

    path_recordings = []
    for audiofile in training_files:
        path_recordings.append(os.path.join(train_folder, audiofile))

    # convert selected label names to integers
    labels_to_index = {
        "Acoustic_guitar": 0,
        "Applause": 1,
        "Bass_drum": 2,
        "Cello": 3,
        "Clarinet": 4,
        "Double_bass": 5,
        "Fart": 6,
        "Fireworks": 7,
        "Flute": 8,
        "Hi-hat": 9,
        "Laughter": 10,
        "Saxophone": 11,
        "Shatter": 12,
        "Snare_drum": 13,
        "Squeak": 14,
        "Tearing": 15,
        "Trumpet": 16,
        "Violin_or_fiddle": 17,
    }

    # encode labels to integers
    get_labels = labels_chosen["label"].replace(labels_to_index).to_list()
    labels_chosen = labels_chosen.reset_index()

    # data is normalized upon loading
    # load dataset
    x_spec, y_number = load_fsdk18(
        path_recordings, labels_chosen, get_labels, feature_type
    )

    nums = list(range(18))
    samples_space = np.geomspace(10, 450, num=6, dtype=int)
    # define path, samples space and number of class combinations
    if feature_type == "melspectrogram":
        prefix = args.m + "_class_mel/"
    elif feature_type == "spectrogram":
        prefix = args.m + "_class/"
    elif feature_type == "mfcc":
        prefix = args.m + "_class_mfcc/"

    # create list of classes with const random seed
    random.Random(5).shuffle(nums)
    classes_space = list(combinations_45(nums, n_classes))

    # scale the data
    x_spec = scale(x_spec.reshape(len(x_spec), -1), axis=1).reshape(len(x_spec), 32, 32)
    y_number = np.array(y_number)

    # need to blance the datasets for each class
    trainx, remainx, trainy, remainy = train_test_split(
        x_spec,
        y_number,
        shuffle=True,
        test_size=0.5,
        stratify=y_number,
    )

    testx, valx, testy, valy = train_test_split(
        remainx,
        remainy,
        shuffle=True,
        test_size=0.5,
        stratify=remainy,
    )

    # 3000 samples, 80% train is 2400 samples, 20% test
    fsdk18_train_images = trainx.reshape(-1, 32 * 32)
    fsdk18_train_labels = trainy.copy()
    # reshape in 2d array
    fsdk18_test_images = testx.reshape(-1, 32 * 32)
    fsdk18_test_labels = testy.copy()
    # validation set
    fsdk18_valid_images = valx.reshape(-1, 32 * 32)
    fsdk18_valid_labels = valy.copy()

    # Start training
    print("Running GBT tuning \n")
    run_GBT()

    print("Running RF tuning \n")
    run_naive_rf()

    print("Running CNN32 tuning \n")
    run_cnn32()

    print("Running CNN32_2l tuning \n")
    run_cnn32_2l()

    print("Running CNN32_5l tuning \n")
    run_cnn32_5l()

    print("Running Resnet tuning \n")
    run_resnet18()

    

