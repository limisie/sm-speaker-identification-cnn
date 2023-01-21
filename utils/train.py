import torch
from torch import nn, optim
from torchvision import datasets, models
from torchvision.models import SqueezeNet1_1_Weights

from consts import CLASSES, EPOCHS, TEST_DATA_ROOT, MODEL_PATH
from ai_utils import device, train_model, data_transform, test_model

m_squeezenet = models.squeezenet1_1(weights=SqueezeNet1_1_Weights.DEFAULT)

for param in m_squeezenet.parameters():
    param.requires_grad = False

m_squeezenet.classifier[1] = nn.Conv2d(512, CLASSES, kernel_size=(1, 1), stride=(1, 1))

m_squeezenet = m_squeezenet.to(device)

model = m_squeezenet
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = None

model, loss, acc = train_model(model,
                               criterion,
                               optimizer,
                               scheduler,
                               num_epochs=EPOCHS)

torch.save(model, f'../models/m_squeezenet_epochs{EPOCHS}.pt')

test_dataset = datasets.ImageFolder(TEST_DATA_ROOT, data_transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
test_size = len(test_dataset)
CLASS_NAMES = test_dataset.classes

model = torch.load(MODEL_PATH, map_location=device)
target_layer = [model.classifier[1]]

labels, predictions, probabilities, probs = test_model(model, test_loader)

print(labels)
print(predictions)
print(probabilities)
print(probs)
