import torch
import torch.nn as nn

class VGG13(nn.Module):
    def __init__(self, in_channels=3, num_classes=2):

        super(VGG13, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # convolutional layers 
        self.conv_layers = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, kernel_size=(3,3), padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3,3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
            nn.Dropout2d(0.20),

            nn.Conv2d(64, 128, kernel_size=(3,3), padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=(3,3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
            nn.Dropout2d(0.20),

            nn.Conv2d(128, 256, kernel_size=(3,3), padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3,3), padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3,3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
            nn.Dropout2d(0.20),

            nn.Conv2d(256, 256, kernel_size=(3,3), padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3,3), padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3,3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
            nn.Dropout2d(0.20)
        )

        # fully connected linear layers
        self.linear_layers = nn.Sequential(
            nn.Linear(256*11*11, 1024),
            nn.ReLU(),
            nn.Dropout2d(0.4),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout2d(0.4),

            # output layer
            nn.Linear(1024, self.num_classes),
        )

        # self.softmax = nn.Softmax(dim=1)

    # forward pass    
    def forward(self, x):
        x = self.conv_layers(x)
        #print(x.shape)
        x = x.view(x.size(0), -1)
        #print(x.shape)
        x = self.linear_layers(x)
        #print(x)
        #print(x.shape)
        # x = self.softmax(x)
        x = nn.functional.softmax(x, dim=1)
        #print(x)
        #print(x.shape)
        return x

'''
if __name__ == '__main__':
    model = VGG13()
    print(model)

    x = torch.randn(1, 3, 180, 180)
    return_value = model(x)
    #print(return_value)
'''


      