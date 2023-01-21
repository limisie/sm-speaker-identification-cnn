import copy
import os
import random
import time

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torchvision import datasets, transforms

from consts import THRESHOLD, CLASS_NAMES, CLASSES, DATA_ROOT, BATCH_SIZE

np.set_printoptions(precision=2, suppress=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor()
])

img_datasets = {x: datasets.ImageFolder(os.path.join(DATA_ROOT, x), data_transform) for x in
                ['train', 'val', 'test']}

data_loaders = {x: torch.utils.data.DataLoader(img_datasets[x],
                                               batch_size=BATCH_SIZE,
                                               shuffle=True,
                                               num_workers=0)
                for x in ['train', 'val', 'test']}

DATASET_SIZES = {x: len(img_datasets[x]) for x in ['train', 'val', 'test']}


def analyze(model_path, data_path):
    model = torch.load(model_path)
    results = get_predictions(model, get_test_dataloader(data_path, data_transform))
    return results


def numpy_to_img(arr):
    arr = arr.transpose((1, 2, 0))
    img = (arr * 255).astype(np.uint8)
    return img


def tensor_to_img(tensor):
    arr = tensor.numpy()
    return numpy_to_img(arr)


def split_dataset(dataset_path, train_split=75, val_split=15, test_split=10):
    labels = [label for label in os.listdir(dataset_path) if not label.startswith('.')]

    try:
        os.mkdir(os.path.join(dataset_path, 'train'))
        os.mkdir(os.path.join(dataset_path, 'val'))
        os.mkdir(os.path.join(dataset_path, 'test'))
    except:
        print('Train, val and test dirs exist')

    for label in labels:
        os.mkdir(os.path.join(dataset_path, 'train', label))
        os.mkdir(os.path.join(dataset_path, 'val', label))
        os.mkdir(os.path.join(dataset_path, 'test', label))

        class_directory = os.path.join(dataset_path, label)
        files = os.listdir(os.path.join(dataset_path, label))

        n = len(files)
        val = int(n // (100 / val_split))
        test = int(n // test_split)

        for filename in random.sample(files, k=val):
            os.rename(os.path.join(class_directory, filename), os.path.join(dataset_path, 'val', label, filename))

        files = os.listdir(class_directory)
        for filename in random.sample(files, k=test):
            os.rename(os.path.join(class_directory, filename), os.path.join(dataset_path, 'test', label, filename))

        files = os.listdir(class_directory)
        for filename in files:
            os.rename(os.path.join(class_directory, filename), os.path.join(dataset_path, 'train', label, filename))

        os.rmdir(class_directory)


def get_test_dataloader(dataset_path, transform):
    test_dataset = datasets.ImageFolder(dataset_path, transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    return test_loader


def train_model(model, criterion, optimizer, data_loaders, scheduler=None, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    losses = {'train': [],
              'val': []}
    accs = {'train': [],
            'val': []}

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in data_loaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train' and scheduler is not None:
                scheduler.step()

            epoch_loss = running_loss / DATASET_SIZES[phase]
            epoch_acc = running_corrects.double() / DATASET_SIZES[phase]

            losses[phase].append(epoch_loss)
            accs[phase].append(epoch_acc.item())

            print(f'{phase} Loss: {epoch_loss} Acc: {epoch_acc}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60}m {time_elapsed % 60}s')
    print(f'Best val Acc: {best_acc}')

    model.load_state_dict(best_model_wts)
    return model, losses, accs


def test_model(model, data_loader, recording=False):
    model = model.to(device)
    model.eval()
    sm = nn.Softmax(dim=1)

    ground_truths = []
    predictions = []
    probabilities = []
    correct_count = 0
    class_probabilities = {x: [] for x in range(CLASSES)}

    for inputs, labels in data_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            out_probs = sm(outputs)

            for inp, prob, label in zip(inputs, out_probs, labels):
                label = label.item()
                probability = prob[label].item()

                ground_truths.append(label)
                class_probabilities[label].append(probability)
                probabilities.append(prob[0].item())
                predictions.append(prob.max(0, keepdim=True).indices.item())

                img = tensor_to_img(inp)
                plt.imshow(img)
                if recording:
                    plt.title(f'prob:{prob.numpy()}\n'
                              f'prediction: {CLASS_NAMES[predictions[-1]]}')
                else:
                    plt.title(f'prob:{prob.numpy()}\n'
                              f'prediction: {CLASS_NAMES[predictions[-1]]}, label: {CLASS_NAMES[label]}')
                plt.show()

                if ground_truths[-1] == predictions[-1]:
                    correct_count += 1

    print(correct_count / len(ground_truths))

    return np.asarray(ground_truths), np.asarray(predictions), np.array(probabilities), np.asarray(class_probabilities)


def get_predictions(model, data_loader):
    model = model.to(device)
    model.eval()
    sm = nn.Softmax(dim=1)

    results = ['\n\nPREDICTIONS: \ntime - class         class probabilities\n']
    time = 0

    for inputs, _ in data_loader:
        inputs = inputs.to(device)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            out_probs = sm(outputs).numpy() * 100

            for inp, prob in zip(inputs, out_probs):
                if np.max(prob) >= THRESHOLD:
                    prediction = CLASS_NAMES[np.argmax(prob, axis=0)]
                else:
                    prediction = 'unknown'

                temp = f'\n{time}s - {prediction}           {prob}'
                results.append(temp)

        time += 0.5

    return results
