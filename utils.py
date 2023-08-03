import os
import time
import torch
from torch import nn
import torch.quantization
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torch import optim
import statistics
import numpy as np
from sklearn.metrics import accuracy_score
from torch.quantization import QuantStub, DeQuantStub


class LeNet5(nn.Module):
    def __init__(self, q=False):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=6,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=6,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=0,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.lin1 = nn.Sequential(
            nn.Linear(400, 120),
            nn.ReLU(),
        )
        self.lin2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU(),
        )
        self.lin3 = nn.Sequential(
            nn.Linear(84, 10),
            nn.ReLU(),
        )
        self.q = q
        if q:
            self.quant = QuantStub()
            self.dequant = DeQuantStub()

    def forward(self, x):
        if self.q:
            x = self.quant(x)
        # CNNs
        x = self.conv1(x)
        x = self.conv2(x)
        # Flatten
        x = x.reshape(x.size(0), -1)
        # Linear Layers
        x = self.lin1(x)
        x = self.lin2(x)
        x = self.lin3(x)
        if self.q:
            x = self.dequant(x)
        return x

def fuse_model(model):
    """
    Fusing model modules can both make the model faster by saving on memory access
    while also improving numerical accuracy.
    :param model: given model to fuse its modules.
    :return: Fuse "conv+Relu" and "linear+Relu" modules from the given LeNet5 model.
    """
    torch.quantization.fuse_modules(model, [["conv1.0", "conv1.1"],
                                            ["conv2.0", "conv2.1"],
                                            ["lin1.0", "lin1.1"],
                                            ["lin2.0", "lin2.1"],
                                            ["lin3.0", "lin3.1"],
                                            ], inplace=True)


def train(model, dataloader, num_epochs, lr=1e-3, device='cpu'):
    """
    Train the given model based on dataloader over a specified number of epochs.
    :param model: ML model to train
    :param dataloader: includes MNIST data to train given model
    :param num_epochs: number of epochs for the training phase
    :param lr: learning rate
    :param device: Either 'cpu' or 'cuda' when GPU is available.
    """
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(num_epochs):
        batch_losses = []
        for _, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            pred = model(images)
            loss = loss_func(pred, labels)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.detach().item())
        epoch_loss = statistics.mean(batch_losses)
        print('Epoch:', epoch, '\tTrain Loss:', epoch_loss)

def test(model, dataloader, device='cpu'):
    """
    Tests the already trained ML model.
    :param model: ML model to test
    :param dataloader: includes MNIST data to test given model
    :param device: Either 'cpu' or 'cuda' when GPU is available.
    :return: Test accuracy for given test data
    """
    model.eval()
    preds = []
    ground_truth = []
    for _, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        _, pred = torch.max(model(images), 1)
        pred = pred.data.cpu()
        preds.extend(pred.detach().numpy())
        ground_truth.extend(labels.detach().numpy())
    accuracy = round(accuracy_score(ground_truth, preds) * 100, 2)
    return accuracy


def prepare_dataset():
    """
    Responsible for downloading, preparing test and train dataloaders (80%-20% ratio) of MNIST dataset
    :return: test and train data in a dataloader format.
    """
    batch_size = 128
    dataset = MNIST(root='./', download=True, train=True, transform=transforms.ToTensor())
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_data, test_data = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_data, batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size, shuffle=False)
    return train_loader, test_loader


def calc_size_of_model(model):
    """
    Calculates ML model's size.
    :param model: ML model
    :return: ML model in KB size.
    """
    torch.save(model.state_dict(), "temp.p")
    model_size = os.path.getsize("temp.p") / 1e3
    # print('Model type: {} - Size (KB): {}'.format(type, model_size))
    os.remove('temp.p')
    return model_size


def calc_time_model_evaluation(model, test_loader):
    """
    Calculates ML model's inference time and accuracy metrics.
    :param model: ML model
    :param test_loader: test data in dataloader format
    :return: Accuracy and elasped inference time metrics
    """
    start = time.time()
    accuracy = test(model, test_loader)
    elapsed_time = time.time() - start
    # print('''Accuracy: {}% - Elapsed time (seconds): {}'''.format(accuracy, elapsed_time))
    return accuracy, elapsed_time


def load_model(quantized_model, model):
    """
    Loads in the weights into an object meant for quantization.
    :param quantized_model: quantized ML model
    :param model: ML-model
    :return: quantized model loaded based on ML model weights.
    """
    state_dict = model.state_dict()
    quantized_model.load_state_dict(state_dict)

def calc_model_metrics(model, test_loader, num_experiments):
    """
    Calculate ML model's size, accuracy and inference time in average over a number of experiments.
    :param model: ML model
    :param test_loader: includes MNIST data to test given model
    :param num_experiments: number of experiments to run
    :return: ML model's average size, accuracy and inference time
    """
    time_evaluation = []
    accuracies = []
    for i in range(num_experiments):
        accuracy, elapsed_time = calc_time_model_evaluation(model, test_loader)
        accuracies.append(accuracy)
        time_evaluation.append(elapsed_time)
    return round(calc_size_of_model(model),2), round(np.mean(time_evaluation), 2), round(np.mean(accuracies), 2)

def calc_dynamic_quant_metrics(model, test_loader, num_experiments):
    """
    Quantized the given ML model based on "dynamic quantization" method
    :param model: ML model
    :param test_loader: includes MNIST data to test given model
    :param num_experiments: number of experiments to run
    :return: Quantized ML model's average size, accuracy and inference time
    """
    quantized_model_size = []
    quantized_time_evaluation = []
    quantized_accuracy = []
    for i in range(num_experiments):
        # create a quantized model instance
        quantized_model = torch.quantization.quantize_dynamic(
            model,  # the original model
            {torch.nn.Linear},  # a set of layers to dynamically quantize
            dtype=torch.qint8)  # the target dtype for quantized weights
        quantized_model_size.append(calc_size_of_model(quantized_model))
        accuracy, elapsed_time = calc_time_model_evaluation(quantized_model, test_loader)
        quantized_accuracy.append(accuracy)
        quantized_time_evaluation.append(elapsed_time)
    return round(np.mean(quantized_model_size), 2), round(np.mean(quantized_time_evaluation), 2), round(np.mean(quantized_accuracy), 2)


def calc_post_training_static_quant_metrics(model, test_loader, quantitation_method, num_experiments):
    """
    Quantized the given ML model based on "post training static quantization" method
    :param model: ML model
    :param test_loader: includes MNIST data to test given model
    :param num_experiments: number of experiments to run
    :return: Quantized ML model's average size, accuracy and inference time
    """
    quantized_model_size = []
    quantized_time_evaluation = []
    quantized_accuracy = []
    for i in range(num_experiments):
        quantized_model = LeNet5(q=True)
        load_model(quantized_model, model)
        fuse_model(quantized_model)
        quantized_model.qconfig = torch.quantization.get_default_qconfig(quantitation_method)
        quantized_model.eval()
        # insert observers
        torch.quantization.prepare(quantized_model, inplace=True)
        # Calibrate the model and collect statistics
        test(quantized_model, test_loader)
        # convert to quantized version
        torch.quantization.convert(quantized_model, inplace=True)
        quantized_model_size.append(calc_size_of_model(quantized_model))
        accuracy, elapsed_time = calc_time_model_evaluation(quantized_model, test_loader)
        quantized_accuracy.append(accuracy)
        quantized_time_evaluation.append(elapsed_time)
    return round(np.mean(quantized_model_size), 2), round(np.mean(quantized_time_evaluation), 2), round(np.mean(quantized_accuracy), 2)


def calc_quant_aware_training_metrics(model, train_loader, test_loader, quantitation_method, num_experiments):
    """
    Quantized the given ML model based on "Quantization aware training" method
    :param model: ML model
    :param test_loader: includes MNIST data to test given model
    :param num_experiments: number of experiments to run
    :return: Quantized ML model's average size, accuracy and inference time
    """
    quantized_model_size = []
    quantized_time_evaluation = []
    quantized_accuracy = []
    for i in range(num_experiments):
        # specify quantization config for QAT
        quantized_model = LeNet5(q=True)
        load_model(quantized_model, model)
        fuse_model(quantized_model)
        quantized_model.qconfig = torch.quantization.get_default_qat_qconfig(quantitation_method)
        # prepare QAT
        torch.quantization.prepare_qat(quantized_model, inplace=True)
        # Retrieve saved quantized model, otherwise train it
        if not os.path.isfile("models/qmodel_32.pth"):
            train(quantized_model, train_loader, num_epochs=10)
            # Save weights
            torch.save(quantized_model.state_dict(), "models/qmodel_32.pth")
        else:
            # Load model
            quantized_model.load_state_dict(torch.load("models/qmodel_32.pth"))
        # convert to quantized version, removing dropout, to check for accuracy on each
        epoch_quantized_model = torch.quantization.convert(quantized_model.eval(), inplace=False)
        quantized_model_size.append(calc_size_of_model(epoch_quantized_model))
        accuracy, elapsed_time = calc_time_model_evaluation(epoch_quantized_model, test_loader)
        quantized_accuracy.append(accuracy)
        quantized_time_evaluation.append(elapsed_time)
    return round(np.mean(quantized_model_size), 2), round(np.mean(quantized_time_evaluation), 2), round(np.mean(quantized_accuracy), 2)
