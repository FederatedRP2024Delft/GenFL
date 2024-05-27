
import collections

import numpy as np
from sklearn.metrics import f1_score
from torch import nn, device, cuda
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch import tensor


# https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer

device = device('cuda' if cuda.is_available() else 'cpu')


def pad_num(k_s):
    pad_per_side = int((k_s - 1) * 0.5)
    return pad_per_side


class SE(nn.Module):
    def __init__(self, cin, ratio):
        super().__init__()
        self.gavg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(cin, int(cin / ratio), bias=False)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(self.fc1.out_features, cin, bias=False)
        self.act2 = nn.Sigmoid()

    def forward(self, x):
        y = x
        x = self.gavg(x)
        x = x.view(-1, x.size()[1])
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = x.view(-1, x.size()[1], 1, 1)
        return x * y


class SE_LN(nn.Module):
    def __init__(self, cin):
        super().__init__()
        self.gavg = nn.AdaptiveAvgPool2d((1, 1))
        self.ln = nn.LayerNorm(cin)
        self.act = nn.Sigmoid()

    def forward(self, x):
        y = x
        x = self.gavg(x)
        x = x.view(-1, x.size(1))
        x = self.ln(x)
        x = self.act(x)
        x = x.view(-1, x.size(1), 1, 1)
        return x * y


class DFSEBV1(nn.Module):
    def __init__(self, cin, dw_s, ratio, is_LN):
        super().__init__()
        self.pw1 = nn.Conv2d(cin, cin, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cin)
        self.act1 = nn.ReLU()
        self.dw1 = nn.Conv2d(cin, cin, dw_s, 1, pad_num(dw_s), groups=cin)
        self.act2 = nn.Hardswish()
        if is_LN:
            self.se1 = SE_LN(cin)
        else:
            self.se1 = SE(cin, 3)

        self.pw2 = nn.Conv2d(cin, cin, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cin)
        self.act3 = nn.ReLU()
        self.dw2 = nn.Conv2d(cin, cin, dw_s, 1, pad_num(dw_s), groups=cin)
        self.act4 = nn.Hardswish()
        if is_LN:
            self.se2 = SE_LN(cin)
        else:
            self.se2 = SE(cin, 3)

    def forward(self, x):
        y = x
        x = self.pw1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.dw1(x)
        x = self.act2(x)
        x = self.se1(x)
        x = x + y

        x = self.pw2(x)
        x = self.bn2(x)
        x = self.act3(x)
        x = self.dw2(x)
        x = self.act4(x)
        x = self.se2(x)
        x = x + y

        return x


class ME(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2, ceil_mode=True)
        self.pw = nn.Conv2d(cin, cout, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x):
        x = self.maxpool(x)
        x = self.pw(x)
        x = self.bn(x)
        return x


class ExquisiteNetV1(nn.Module):
    def __init__(self, class_num, img_channels):
        super().__init__()
        self.features = nn.Sequential(
            collections.OrderedDict([
                ('ME1', ME(img_channels, 12)),
                ('DFSEB1', DFSEBV1(12, 3, 3, False)),

                ('ME2', ME(12, 50)),
                ('DFSEB2', DFSEBV1(50, 3, 3, False)),

                ('ME3', ME(50, 100)),
                ('DFSEB3', DFSEBV1(100, 3, 3, False)),

                ('ME4', ME(100, 200)),
                ('DFSEB4', DFSEBV1(200, 3, 3, False)),

                ('ME5', ME(200, 350)),
                ('DFSEB5', DFSEBV1(350, 3, 3, False)),

                ('conv', nn.Conv2d(350, 640, 1, 1)),
                ('act', nn.Hardswish())
            ])
        )
        self.gavg = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(0.2)
        self.fc = nn.Linear(640, class_num)

    def forward(self, x):
        x = self.features(x)
        x = self.gavg(x)
        x = self.drop(x)
        x = x.view(-1, x.size()[1])
        x = self.fc(x)
        # x = F.log_softmax(x, dim=1)
        return x

    def train_model(
            self,
            training_data,
            testing_data,
            batch_size,
            learning_rate,
            epochs
    ):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.parameters(), lr=learning_rate)
        training_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            self.train()
            for input, labels in training_dataloader:
                input, labels = input.to(device), labels.to(device)
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.forward(input)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            print("Epoch done: ", epoch + 1)
            print("Training loss: ", loss.item())
            _, loss, _, _ = self.test_inference(testing_data, batch_size)
            print("Testing loss: ", loss)

    def train_model_syn_image(
            self,
            input_li,
            labels_li,
            epochs=5,
            learning_rate=0.01
    ):
        """
        input_li, label_li are list of tensors.

        a tensor from input_li should have shape (<batch> 1, 28, 28)
        a tensor from label_li should have shape (<batch>)
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            self.train()
            for i in range(len(input_li)):
                input = input_li[i]
                labels = labels_li[i]

                input = input.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.forward(input)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            print("Epoch done: ", epoch + 1)
            print("Training loss: ", loss.item())


    def test_inference(self, test_dataset, batch_size):
        # model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        pred_labels_list = []
        true_labels_list = []

        criterion = nn.NLLLoss().to(device)
        testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = self.forward(images)
            batch_loss = criterion(outputs, labels)

            # accumulate average of batch losses
            loss += batch_loss.item() / len(testloader)

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

            pred_labels_list.append(pred_labels.cpu().numpy())
            true_labels_list.append(labels.cpu().numpy())

        pred_labels_all = np.concatenate(pred_labels_list, axis=0)
        true_labels_all = np.concatenate(true_labels_list, axis=0)
        f1_macro = f1_score(true_labels_all, pred_labels_all, average='macro')
        f1_micro = f1_score(true_labels_all, pred_labels_all, average='micro')
        accuracy = correct / total
        return accuracy, loss, f1_macro, f1_micro

    def generate_labels(
            self,
            input: tensor
    ) -> tensor:
        labels = []
        for img in input:
            outputs = self.forward(img)
            _, predicted = torch.max(outputs.data, 1)
            labels.append(predicted)
        return torch.stack(labels)

    def test_model_syn_img_label(
            self,
            testing_data: tensor,
            labels: tensor
    ) -> float:
        wrong = 0
        correct = 0
        total = 0
        for i, input in enumerate(testing_data):
            input = input.to(device)
            outputs = self.forward(input)
            _, predicted = torch.max(outputs.data, 1)
            _, label = torch.max(labels[i], 0)
            if predicted != label:
                wrong += 1
            else:
                correct += 1
            total += 1
        # print("Wrong count: ", wrong)
        return correct / total