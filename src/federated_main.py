#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
#python federated_main.py --model=vae --dataset=mnist --gpu=cuda:0 --iid=2 --epochs=30 --dirichlet=0.4 --frac=1.0 --num_users=10 --local_ep=10

import os
import copy
import time
import pickle
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from tensorboardX import SummaryWriter
from options import args_parser
from update import LocalUpdate
from utils import get_dataset, exp_details, fed_avg
from vae.mnist_vae import ConditionalVae
from impute import impute_cvae_naive
from tqdm import tqdm
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from image_classifier.exq_net_v1 import ExquisiteNetV1
import torch
from torch import nn, optim

if __name__ == '__main__':
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')

    args = args_parser()
    exp_details(args)

    if args.gpu:
        torch.cuda.set_device(args.gpu)
    device = 'cuda' if args.gpu else 'cpu'

    # load dataset and user groups
    training_dataset, testing_dataset, user_groups = get_dataset(args)


    # BUILD MODEL
    if args.model == 'cvae':
        global_model = ConditionalVae(dim_encoding=3)
    else:
        global_model = ExquisiteNetV1(class_num=10, img_channels=1)

    model_dict_path = f"../models/federated_{args.model}_{args.dataset}_{args.iid}_{args.dirichlet}_{args.epochs}_{args.local_ep}_{args.num_users}.pt"

    if os.path.exists(model_dict_path):
        global_model.load_state_dict(torch.load(model_dict_path))
    else:
        # Set the model to train and send it to device.
        global_model.to(device)
        global_model.train()

        # if set to pretrain and classifier, pre-train with synthetic data on server
        if args.pretrain == 'True' and args.model == 'exq':
            gen_model_path = f"../models/federated_cvae_{args.dataset}_{args.iid}_{args.dirichlet}_20_{args.local_ep}_{args.num_users}.pt"
            gen_model = ConditionalVae(dim_encoding=3)
            gen_model.load_state_dict(torch.load(gen_model_path))

            gen_train_dataset = impute_cvae_naive(k=60000, trained_cvae=gen_model, initial_dataset=torch.tensor([]))

            batch_size = 32
            learning_rate = 0.001
            epochs = 5

            # train classifier on gen data
            train_loader = DataLoader(gen_train_dataset, batch_size=batch_size, shuffle=True)

            # Define the loss function and the optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(global_model.parameters(), lr=learning_rate)

            # Number of epochs to train the model
            for epoch in tqdm(range(epochs)):
                for data, target in train_loader:
                    data, target = data.to(device), target.to(device)

                    # Clear the gradients of all optimized variables
                    optimizer.zero_grad()

                    # Forward pass: compute predicted outputs by passing inputs to the model
                    output = global_model(data)

                    # Calculate the loss
                    loss = criterion(output, target)

                    # Backward pass: compute gradient of the loss with respect to model parameters
                    loss.backward()

                    # Perform single optimization step (parameter update)
                    optimizer.step()

        # copy weights
        global_weights = global_model.state_dict()

        # Training
        train_losses = []
        train_losses_per_client = np.zeros((args.num_users, args.epochs))

        # Test
        test_losses = []
        test_weighted_accuracies = []
        test_losses_per_client = np.zeros((args.num_users, args.epochs))
        test_accuracy_per_client = np.zeros((args.num_users, args.epochs))

        for epoch in tqdm(range(args.epochs)):
            local_weights, local_losses = [], []

            global_model.train()

            m = max(int(args.frac * args.num_users), 1)
            idxs_users = np.random.choice(range(args.num_users), m, replace=False)
            dataset_size_per_client = [len(user_groups[i]) for i in idxs_users]

            print("Client indices")
            print(idxs_users)
            print()
            print("Dataset size per client")
            print(dataset_size_per_client)
            print()
            ratio_per_client = [0] * args.num_users
            for i in idxs_users:
                ratio_per_client[i] = (dataset_size_per_client[i] / sum(dataset_size_per_client))

            print("Ratio per client")
            print(ratio_per_client)

            print("Total dataset")
            print(sum(dataset_size_per_client))

            # all test data will be 0.1 of each client's total dataset
            print("Ratios sum")
            print(sum(ratio_per_client))

            for idx in idxs_users:
                local_model = LocalUpdate(args=args, dataset=training_dataset,
                                          idxs=user_groups[idx], logger=logger)

                # create a new instance of the same model
                if args.model == 'cvae':
                    model_copy = type(global_model)(3).to(device)
                else:
                    model_copy = type(global_model)(10, 1).to(device)
                model_copy.load_state_dict(global_model.state_dict())

                w, loss = local_model.update_weights(model=model_copy, global_round=epoch)
                local_weights.append(copy.deepcopy(w))
                train_losses_per_client[idx][epoch] = loss
                # print(f"actual loss: {loss}")
                if np.isnan(loss):
                    # print("loss was nan!!!!!!!!!!!!!!!")
                    loss = local_losses[-1] if len(local_losses) > 0 else 0
                local_losses.append(copy.deepcopy(loss))

            # update global weights
            global_weights = fed_avg(local_weights, dataset_size_per_client)

            # update global weights
            global_model.load_state_dict(global_weights)

            loss_avg = sum(local_losses) / len(local_losses)
            train_losses.append(loss_avg)

            # inference for each client
            list_loss, list_weighted_acc, list_f1_score = [], [], []
            global_model.eval()
            for idx in idxs_users:
                local_model = LocalUpdate(args=args, dataset=training_dataset,
                                          idxs=user_groups[idx], logger=logger)

                acc, loss = local_model.inference(model=global_model)
                test_losses_per_client[idx][epoch] = loss
                test_accuracy_per_client[idx][epoch] = acc
                list_loss.append(loss)
                list_weighted_acc.append(acc * ratio_per_client[idx])
                # list_f1_score.append()

            test_losses.append(sum(list_loss) / len(list_loss))
            total_weighted_accuracies = sum(list_weighted_acc)
            test_weighted_accuracies.append(total_weighted_accuracies)
            # test_weighted_accuracies.append(total_weighted_accuracies)


            print(f' \nAvg Training Stats after {epoch+1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_losses))}')
            print('Test Accuracy: {:.2f}% \n'.format(100 * total_weighted_accuracies))

        print("Train losses per communication round: ", train_losses)
        print("Test losses per communication round: ", test_losses)
        print("Test weighted accuracies per communication round: ", test_weighted_accuracies)
        # print("Test weighted F1 scores per communication round: ", test_weighted_f1_scores)

        print("Test losses per communication round for each client: ", test_losses_per_client)
        print("Test accuracies per communication round for each client: ", test_accuracy_per_client)
        # print("Test F1-score per communication round for each client: ", test_accuracies_per_client)

        torch.save(global_model.state_dict(), model_dict_path)


    # train classifier on syn data and test on 70000 real data set
    if args.model == 'cvae':
        # generate data to train classifier to determine CAS score
        final_training_data = impute_cvae_naive(k=60000, trained_cvae=global_model, initial_dataset=torch.tensor([]))
        final_testing_data = torch.utils.data.ConcatDataset([training_dataset, testing_dataset])

        print("Final testing dataset size: ", len(final_testing_data))

        # train classifier on gen data
        batch_size = 32
        learning_rate = 0.001
        epochs = 20

        train_loader = torch.utils.data.DataLoader(final_training_data, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(final_testing_data, batch_size=batch_size, shuffle=True)

        classifier = ExquisiteNetV1(class_num=10, img_channels=1).to(device)

        # Define the loss function and the optimizer
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(classifier.parameters(), lr=learning_rate)

        # Number of epochs to train the model
        train_losses = []
        test_losses = []
        f1_scores = []
        cas_scores = []
        correct_predictions = 0
        total_predictions = 0
        for epoch in tqdm(range(epochs)):
            train_loss = 0.0
            pred_labels = []
            actual_labels = []
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)

                # Clear the gradients of all optimized variables
                optimizer.zero_grad()

                # Forward pass: compute predicted outputs by passing inputs to the model
                output = classifier(data)
                pred_labels.append(output.argmax(dim=1))
                actual_labels.append(target)

                # Calculate the loss
                loss = criterion(output, target)

                # Backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()

                # Perform single optimization step (parameter update)
                optimizer.step()

                # Update running training loss
                train_loss += loss.item() * data.size(0)

            # Switch to evaluation mode
            classifier.eval()
            with torch.no_grad():
                test_loss = 0.0
                test_pred_labels = []
                test_actual_labels = []
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    output = classifier(data)
                    loss = criterion(output, target)
                    test_loss += loss.item() * data.size(0)
                    test_pred_labels.append(output.argmax(dim=1))
                    test_actual_labels.append(target)
                    # Compare with actual classes
                    total_predictions += output.argmax(dim=1).size(0)
                    # correct_predictions += (predicted == labels).sum().item()
                    correct_predictions += (output.argmax(dim=1) == target).sum().item()

            # Compute average test loss
            train_loss = train_loss / len(train_loader.dataset)
            test_loss = test_loss / len(test_loader.dataset)
            test_losses.append(test_loss)
            train_losses.append(train_loss)

            # Calculate F1 score for the test data
            test_pred_labels = torch.cat(test_pred_labels).to('cpu').numpy()
            test_actual_labels = torch.cat(test_actual_labels).to('cpu').numpy()
            test_f1_score = f1_score(test_actual_labels, test_pred_labels, average='macro')
            f1_scores.append(test_f1_score)
            accuracy = correct_predictions / total_predictions
            cas_scores.append(accuracy)

            # test classifier with real testing data per epoch
            print(f'CAS: {accuracy * 100}%')
            print('Epoch: {} \tTraining Loss: {:.6f} \t Test Loss: {:.6f} \tF1 Test Macro: {:.6f}'.format(
                epoch + 1,
                train_loss,
                test_loss,
                test_f1_score
            ))

        # final CAS score testing on 70000
        correct_predictions = 0
        total_predictions = 0
        with torch.no_grad():
            for data, labels in test_loader:
                data, labels = data.to(device), labels.to(device)

                # Pass the data to the model
                outputs = classifier(data)

                # Get the predicted class with the highest score
                _, predicted = torch.max(outputs.data, 1)

                # Compare with actual classes
                total_predictions += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()

        accuracy = correct_predictions / total_predictions
        print(f'CAS: {accuracy * 100}%')

        print("Train losses: ", train_losses)
        print("Test losses: ", test_losses)
        print("F1 scores: ", f1_scores)
        print("CAS scores: ", cas_scores)