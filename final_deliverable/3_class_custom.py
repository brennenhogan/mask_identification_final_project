import pandas as pd 
import os
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision
from torchvision import torch,datasets,transforms,models
from torch.utils.data import Dataset,DataLoader
import torch
from torch import nn
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD
from PIL import Image
import numpy as np 

#Colors cooresponding to the labels
class_color   = {"face_no_mask":"r","face_with_mask":"g","face_with_mask_incorrect":"b"}
label_to_ints = {"face_no_mask":0,"face_with_mask":1,"face_with_mask_incorrect":2}
ints_to_label = {0:"face_no_mask",1:"face_with_mask",2:"face_with_mask_incorrect"}

#Network with 3 hidden layers and sigmoid activation function
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.cnn_layers = Sequential(
            # Defining a 2D convolution layer
            Conv2d(3, 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer
            Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
        )

        self.linear_layers = Sequential(
            Linear(12544, 10)
        )

    # Defining the forward pass    
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x

#Class used to crop the images and standardize them to the same aspect ratio
class ImageStandardizer(Dataset): 
    def __init__(self,dataframe,file_path,transform=None):
        self.annotation = dataframe
        self.file_path   = file_path
        self.transform  = transform      
        
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, i):
        image_path      = os.path.join(self.file_path,self.annotation.iloc[i, 0])
        cropped_image   = Image.open(image_path).crop((self.annotation.iloc[i, 1:5]))
        label           = torch.tensor(label_to_ints[self.annotation.iloc[i, 5]])
    
        #Resizes the cropped image to 224 x 224
        if self.transform:
            image=self.transform(cropped_image)
            return(image,label)

#Class used to crop the images and standardize them to the same aspect ratio
class UserData(Dataset): 
    def __init__(self,file_path,image_path,transform=None):
        self.file_path   = file_path
        self.image_path  = image_path
        self.transform   = transform      
        
    def __len__(self):
        return 1
    
    def __getitem__(self, i):
        image_path      = os.path.join(self.file_path,self.image_path)
        cropped_image = "test.jpg"
        try:
            cropped_image   = Image.open(image_path)
        except:
            print("File does not exist")
    
        #Resizes the cropped image to 224 x 224
        if self.transform:
            image=self.transform(cropped_image)
            return(image)

# creating the function to visualize images and draw a box around the face
def visualize_guess(image_name, label):
    #Reads in the image and creates the plot
    image=plt.imread(os.path.join('../mask_data/dataset/masks/images',image_name))
    fig,ax=plt.subplots(1)
    ax.axis('off')
    fig.set_size_inches(20,10)
    ax.imshow(image)

    #Add the text box
    ax.text(0, 0, label, style='italic',bbox={'facecolor': class_color[label], 'alpha': 0.75})

    #Plots the image with the label
    imgplot = plt.imshow(image)
    plt.show()

def run_model_test(model, training_data, testing_data):
    #Trains the model for 5 epochs in batches of 10
    epochs = 5
    training_loss = []

    for epoch in range(epochs):
        train_loss = 0
        
        #Work with the training data in batches of 10 to make program more responsive
        for batch, (data,target) in enumerate(training_data):

            optimizer.zero_grad()
            output=model(data)
            loss=criterion(output,target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
            if batch%10==9:
                print("Epoch: {}, Total Batches: {}\nAverage Training Loss: {}".format(epoch+1,batch+1,train_loss/10))
                training_loss.append(train_loss) 
                train_loss=0.0
    
    print("Total training loss {}".format(training_loss))

    #Test the model using the test dataset
    test_loss = 0
    correct = 0
    effective_correct = 0
    attempted = 0
    model.eval()

    for data,target in testing_data:
      
        output = None
  
        with torch.no_grad():
            output = model(data)

        processed_output = post_processing(output)
        
        for i in range(len(processed_output)):
            attempted += 1
            if processed_output[i] == target[i]:
                correct += 1
                effective_correct += 1
            elif (target[i] == 1 or target[i] == 2) and (processed_output[i] == 1 or processed_output[i] == 2):
                effective_correct += 1
        
        print("Predictions: {}".format(processed_output))
        print("Target: {}".format(target))
        print("-----")
                
        loss = criterion(output,target)
        test_loss += loss.item()
        
    avg_loss=test_loss/attempted

    print("Average total loss is {:.6f}".format(avg_loss))
    print("{} correct predictions out of {} total images".format(correct, attempted))
    print("{} effective correct predictions out of {} total images".format(effective_correct, attempted))

def run_model_demo(model, training_data, image_path, image_transformer):
    #Trains the model for 5 epochs in batches of 10
    epochs = 5
    training_loss = []

    for epoch in range(epochs):
        train_loss = 0
        
        #Work with the training data in batches of 10 to make program more responsive
        for batch, (data,target) in enumerate(training_data):

            optimizer.zero_grad()
            output=model(data)
            loss=criterion(output,target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
            if batch%10==9:
                print("Epoch: {}, Total Batches: {}\nAverage Training Loss: {}".format(epoch+1,batch+1,train_loss/10))
                training_loss.append(train_loss) 
                train_loss=0.0
    
    print("Total training loss {}".format(training_loss))

    #Run the model demo
    print("The model is trained and ready to make predictions!")

    model.eval()

    while True:
        image = input("Enter the file name of the image to be evaluated or E to exit: ")
        
        if image == "E":
            print("Thanks for participating!")
            break

        user_image = UserData(image_path, image, image_transformer)
        user_data    = DataLoader(user_image)

        for data in user_data:
            output = None

            with torch.no_grad():
                output = model(data)

            processed_output = post_processing(output)
            guess = ints_to_label[processed_output.item()]
            print("The guess is: {}".format(guess))
            visualize_guess(image, guess)

#Simple post processing for getting the rounded values
def post_processing(output): 
    probs, classes = output.topk(1, dim=1)
    return classes

if __name__ == '__main__':
    mode = input("Enter your mode (T for test, D for demo): ")
    lr = float(input("Type in a learning rate (0.003 is the reccomended minimum value): "))

    #Read in training data
    train = pd.read_csv('../3_class.csv')

    print("dataset spread")
    print(train.classname.value_counts())
    print("-----")

    #Training data makes up 75% of the dataset and testing data makes up the remaining 25%
    train_size = int(len(train)*0.75)
    test_size  = len(train)-train_size
    
    print("Train size: {}".format(train_size))
    print("Test size: {}".format(test_size))

    image_transformer = transforms.Compose([transforms.Resize((224,224)),
                                 transforms.RandomCrop((224,224)),
                                 transforms.ToTensor()])

    image_path = os.path.join("../mask_data/dataset/masks/images")

    dataset = ImageStandardizer(train, image_path, image_transformer)

    batch_size = 10

    trainset,testset = torch.utils.data.random_split(dataset,[train_size,test_size])
    training_data    = DataLoader(trainset, batch_size, True)
    testing_data     = DataLoader(testset, batch_size, True)

    dataiter = iter(training_data)
    images,labels = dataiter.next()

    #Download the pre-trained rsnet facial recognition model
    model = Network()

    #Sets the loss function
    criterion = nn.CrossEntropyLoss()

    #Sets the model learning rate
    optimizer = torch.optim.SGD(model.parameters(),lr=lr)

    if mode == "T":
        run_model_test(model, training_data, testing_data)
    elif mode == "D":
        run_model_demo(model, training_data, image_path, image_transformer)