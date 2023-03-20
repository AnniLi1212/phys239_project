
"""
Created on Sun Mar 19 17:33:28 2023

@author: daniel
"""
import torch 
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# TODO: check numbers in arguments
class CNNModel(nn.Module):
    def __init__(self, input_shape):#, num_classes):
        super(CNNModel, self).__init__()
        print(input_shape)
        self.conv1 = nn.Conv2d(input_shape[0], 64, kernel_size=7, stride=2, padding=3) #THIS MAY NOT BE CORRECT INPUT SHAPE
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.lrn1 = nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)  #64, 196
        self.lrn2 = nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Inception modules
        self.inc1 = Inception_Module(64, 'inc1') #196
        self.inc2 = Inception_Module(256, 'inc2')
        
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # More inception modules
        self.inc3 = Inception_Module(256, 'inc3')  #THIS MAY NOT BE CORRECT INPUT SHAPE, recplaced 480->256
        
        self.pool4 = nn.AdaptiveAvgPool2d((1, 1))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.lrn1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.lrn2(x)
        x = self.pool2(x)

        # Inception modules
        x = self.inc1(x)
        x = self.inc2(x)

        x = self.pool3(x)

        # More inception modules
        x = self.inc3(x)

        x = self.pool4(x)
        x = x.view(x.size(0), -1)
        x = self.softmax(x)
        return x

class Inception_Module(nn.Module):
    def __init__(self, in_channels, name):
        super(Inception_Module, self).__init__()
        
        self.conv_a1 = nn.Conv2d(in_channels, 64, kernel_size=1)
        self.conv_b1 = nn.Conv2d(in_channels, 96, kernel_size=1)
        self.conv_c1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.pool_d1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        
        self.conv_b2 = nn.Conv2d(96, 128, kernel_size=3, padding=1)
        self.conv_c2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
        self.conv_d2 = nn.Conv2d(in_channels, 32, kernel_size=1)

    def forward(self, x):
        a1 = self.conv_a1(x)
        b1 = self.conv_b1(x)
        c1 = self.conv_c1(x)
        d1 = self.pool_d1(x)
        
        b2 = self.conv_b2(b1)
        c2 = self.conv_c2(c1)
        d2 = self.conv_d2(d1)

        output = torch.cat((a1, b2, c2, d2), dim=1)
        return output
    
# Define the Trainer class
class Trainer:
    def __init__(self, model, train_loader, test_loader, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device

    def train(self, epochs, learning_rate):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)

        for epoch in range(epochs):
            running_loss = 0.0
            for i, data in enumerate(self.train_loader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            print(f'Epoch {epoch + 1}, Loss: {running_loss / (i + 1)}')

    def test(self):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in self.test_loader:
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print('Accuracy: %d %%' % (100 * correct / total))
        
        
# Load the dataset (MNIST)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=100, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=2)

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Instantiate the model and trainer
model = CNNModel(train_set[0][0].shape)
trainer = Trainer(model, train_loader, test_loader, device)

# Train and test the model
trainer.train(epochs=10, learning_rate=0.001)
trainer.test()