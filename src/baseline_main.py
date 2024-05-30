import os

from utils import get_dataset
from vae.mnist_vae import ConditionalVae
from impute import impute_cvae_naive
from tqdm import tqdm
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from image_classifier.exq_net_v1 import ExquisiteNetV1
import torch
from torch import nn, optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if __name__ == '__main__':
    # binarize the data
    class args:
        def __init__(self):
            self.num_channels = 1
            self.iid = 1
            self.num_classes = 10
            self.num_users = 10
            self.dataset = 'fmnist'

    training_data, testing_data, user_groups = get_dataset(args())

    model = "cvae"
    dataset = "fmnist"
    batch_size = 32
    epochs = 20
    learning_rate = 0.001

    model_path = f"../models/local_{model}_{dataset}_{batch_size}_{epochs}_{learning_rate}.pt"

    if os.path.exists(model_path):
        cvae_model = torch.load(model_path)
    else:
        cvae = ConditionalVae(dim_encoding=3).to(device)

        # try with model sigma
        cvae_model, vae_loss_li, kl_loss_li, reg_loss_li = cvae.train_model(
            training_data=training_data,
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate
        )
        torch.save(cvae_model, model_path)

    # generate synthetic data and 70000 real testing dataset
    gen_train_dataset = impute_cvae_naive(k=60000, trained_cvae = cvae_model, initial_dataset = torch.tensor([]))
    final_testing_data = torch.utils.data.ConcatDataset([training_data, testing_data])
    print(len(final_testing_data))

    # train classifier on gen data and test on entire 70000 real dataset
    model = "exq_v1"
    dataset = "fmnist"
    batch_size = 32
    learning_rate = 0.001
    epochs = 20

    train_loader = DataLoader(gen_train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(final_testing_data, batch_size=batch_size, shuffle=True)

    model_path = f"../models/local_{model}_{dataset}_{batch_size}_{epochs}_{learning_rate}.pt"

    classifier = ExquisiteNetV1(class_num=10, img_channels=1).to(device)

    # Define the loss function and the optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)

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

        print(f'CAS score: {accuracy * 100}%')
        print('Epoch: {} \tTraining Loss: {:.6f} \t Test Loss: {:.6f} \tF1 Test Macro: {:.6f}'.format(
            epoch + 1,
            train_loss,
            test_loss,
            test_f1_score
        ))

    print("Train losses: ", train_losses)
    print("Test losses: ", test_losses)
    print("F1 scores: ", f1_scores)
    print("CAS scores: ", cas_scores)

    # torch.save(classifier, model_path)