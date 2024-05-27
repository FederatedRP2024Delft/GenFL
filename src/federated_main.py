#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
#python federated_main.py --model=vae --dataset=mnist --gpu=cuda:0 --iid=2 --epochs=30 --dirichlet=0.4 --frac=1.0 --num_users=10 --local_ep=10

# only for cvae

import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter
from sklearn.metrics import f1_score

from options import args_parser
from vae.mnist_vae import ConditionalVae
from update import LocalUpdate
from image_classifier.exq_net_v1 import ExquisiteNetV1
from utils import get_dataset, exp_details, fed_avg
from impute import impute_cvae_naive
import torchvision

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
    training_data, testing_data, user_groups = get_dataset(args)

    # BUILD MODEL
    if args.model == 'cvae':
        global_model = ConditionalVae(dim_encoding=3)
    else:
        exit('Error: unrecognized model')

    model_dict_path = f"../models/federated_{args.model}_{args.dataset}_{args.iid}_{args.dirichlet}_{args.epochs}_{args.local_ep}_{args.num_users}.pt"
    if os.path.exists(model_dict_path):
        torch.load(model_dict_path)
        global_model.load_state_dict(torch.load(model_dict_path))
    else:
        # Set the model to train and send it to device.
        global_model.to(device)
        global_model.train()

        # copy weights
        global_weights = global_model.state_dict()

        # Training
        train_losses, train_accuracies = [], []
        test_losses, test_accuracies = [], []
        val_acc_list, net_list = [], []
        cv_loss, cv_acc = [], []
        print_every = 2
        val_loss_pre, counter = 0, 0
        test_losses_per_client, test_accuracies_per_client = np.zeros((args.num_users, args.epochs)), np.zeros((args.num_users, args.epochs))
        train_losses_per_client = np.zeros((args.num_users, args.epochs))

        for epoch in tqdm(range(args.epochs)):
            local_weights, local_losses = [], []

            global_model.train()
            m = max(int(args.frac * args.num_users), 1)
            idxs_users = np.random.choice(range(args.num_users), m, replace=False)
            dataset_size_per_client = [len(user_groups[i]) for i in idxs_users]
            for idx in idxs_users:
                local_model = LocalUpdate(args=args, dataset=training_data,
                                          idxs=user_groups[idx], logger=logger)

                model_copy = type(global_model)(3)  # create a new instance of the same model
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
            # print("sum local losses: ", sum(local_losses))
            # print("Num of local losses: ", len(local_losses))
            train_losses.append(loss_avg)

            # Calculate avg training accuracy over all users at every epoch
            list_acc, list_loss = [], []
            global_model.eval()
            for c in range(args.num_users):
                local_model = LocalUpdate(args=args, dataset=training_data,
                                          idxs=user_groups[c], logger=logger)
                acc, loss = local_model.inference(model=global_model)
                list_acc.append(acc)
                test_losses_per_client[c][epoch] = loss
                test_accuracies_per_client[c][epoch] = acc
                # print("Accuracy: ", acc)
                # print("Loss: ", loss)
            train_accuracies.append(sum(list_acc) / len(list_acc))

            # print(f"IID data total communication rounds {i} accuracy: ", accuracy)
            #
            # print global training loss after every 'i' rounds
            # if (epoch+1) % print_every == 0:
            #     print(f' \nAvg Training Stats after {epoch+1} global rounds:')
            #     print(f'Training Loss : {np.mean(np.array(train_loss))}')
            #     print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))

        torch.save(global_model.state_dict(), model_dict_path)

    # generate data to train classifier on
    gen_dataset = impute_cvae_naive(k=60000, trained_cvae=global_model, initial_dataset=torch.tensor([]))

    # train classifier on gen data
    model = "exq_v1"
    dataset = args.dataset
    batch_size = 32
    learning_rate = 0.001
    epochs = 10

    data_dir = f'../data/{dataset}/'
    apply_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda x: torch.round(x))])

    train_loader = torch.utils.data.DataLoader(gen_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testing_data, batch_size=batch_size, shuffle=True)

    classifier = ExquisiteNetV1(class_num=10, img_channels=1).to(device)

    # Define the loss function and the optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=learning_rate)

    # Number of epochs to train the model
    train_losses = []
    test_losses = []
    f1_scores = []
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

        print(f'Accuracy: {accuracy * 100}%')
        print('Epoch: {} \tTraining Loss: {:.6f} \t Test Loss: {:.6f} \tF1 Test Macro: {:.6f}'.format(
            epoch + 1,
            train_loss,
            test_loss,
            test_f1_score
        ))

    # Saving the objects train_loss and train_accuracy:
    file_name = '../objects/cvae_{}_{}_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'.\
        format(args.num_users, args.dirichlet, args.dataset, args.model, args.epochs, args.frac, args.iid,
               args.local_ep, args.local_bs)

    with open(file_name, 'wb') as f:
        pickle.dump([train_losses_per_client, test_losses_per_client, test_accuracies_per_client], f)

    # test classifier with real testing data
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

    print(f'Accuracy: {accuracy * 100}%')


    # print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

    #
    # # PLOTTING (optional)
    # import matplotlib
    # import matplotlib.pyplot as plt
    # matplotlib.use('Agg')
    #
    # # Plot Loss curve
    # plt.figure()
    # plt.title('Training Loss vs Communication rounds')
    # plt.plot(range(len(train_loss)), train_loss, color='r')
    # plt.ylabel('Training loss')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('/home/neo/projects/FederatedImputation/save/cvae_{}_{}_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_loss.png'.
    #             format(args.num_users, args.dirichlet, args.dataset, args.model, args.epochs, args.frac,
    #                    args.iid, args.local_ep, args.local_bs))
    #
    # # Plot Average Accuracy vs Communication rounds
    # plt.figure()
    # plt.title('Average Accuracy vs Communication rounds')
    # plt.plot(range(len(train_accuracy)), train_accuracy, color='k')
    # plt.ylabel('Average Accuracy')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('/home/neo/projects/FederatedImputation/save/cvae_{}_{}_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_acc.png'.
    #             format(args.num_users, args.dirichlet, args.dataset, args.model, args.epochs, args.frac,
    #                    args.iid, args.local_ep, args.local_bs))
