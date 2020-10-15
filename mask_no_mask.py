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
from PIL import Image
import numpy as np 
import ssl

#Certificate for downloading the pretrained model
ssl._create_default_https_context = ssl._create_unverified_context

#Boolean which controls if the images are displayed or not
SHOW_IMAGES = False

#Colors cooresponding to the labels
class_color={"face_no_mask":"r","face_with_mask":"g"}
label_to_ints={"face_no_mask":0,"face_with_mask":1}
ints_to_label={0:"face_no_mask",1:"face_with_mask"}

# creating the function to visualize images and draw a box around the face
def visualize_training(image_name):
    #Reads in the image and creates the plot
    image=plt.imread(os.path.join('./mask_data/dataset/masks/images',image_name))
    temp=train[train.name==image_name]
    fig,ax=plt.subplots(1)
    ax.axis('off')
    fig.set_size_inches(20,10)
    ax.imshow(image)
    for i in range(len(temp)):
        #Grab the x and y bounds of the face from the training data
        x1,x2,y1,y2=temp.values[i][1:5]
        #Draw the rectange around the face
        patch=patches.Rectangle((x1,x2),y1-x1,y2-x2,linewidth=2, 
                                edgecolor=class_color[temp.values[i][5]],facecolor="none",)
        #Add the text box
        ax.text(x1, x2, temp.values[i][5], style='italic',bbox={'facecolor': 'white', 'alpha': 0.75})
        ax.add_patch(patch)

    #Plots the image with the label
    imgplot = plt.imshow(image)
    plt.show()

# creating the function to visualize images and draw a box around the face
def visualize_guess(image_name, labels):
    #Reads in the image and creates the plot
    image=plt.imread(os.path.join('./mask_data/dataset/masks/images',image_name))
    temp=train[train.name==image_name]
    fig,ax=plt.subplots(1)
    ax.axis('off')
    fig.set_size_inches(20,10)
    ax.imshow(image)
    for i in range(len(temp)):

        #Grab the x and y bounds of the face from the training data
        x1,x2,y1,y2=temp.values[i][1:5]
        #Draw the rectange around the face
        patch=patches.Rectangle((x1,x2),y1-x1,y2-x2,linewidth=2, 
                                edgecolor=class_color[labels[i]],facecolor="none",)
        #Add the text box
        ax.text(x1, x2, labels[i], style='italic',bbox={'facecolor': 'white', 'alpha': 0.75})
        ax.add_patch(patch)

    #Plots the image with the label
    imgplot = plt.imshow(image)
    plt.show()

def train_model(model, training_data):
    epochs=3
    training_loss=[]

    for epoch in range(epochs):
        train_loss=0
        
        #Work with the training data in batches of 10 to make program more responsive
        for batch,(data,target) in enumerate(training_data):

            optimizer.zero_grad()
            output=model(data)
            loss=criterion(output,target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
            if batch%10==9:
                print("Epoch: {}, Total Images: {}\nAverage Training Loss: {}".format(epoch+1,batch+1,train_loss/10))
                training_loss.append(train_loss) 
                train_loss=0.0
    
    print(training_loss)

#Class used to crop the images and standardize them to the same aspect ratio
class ImageStandardizer(Dataset): 
    def __init__(self,dataframe,root_dir,transform=None):
        self.annotation=dataframe
        self.root_dir=root_dir
        self.transform=transform      
        
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self,index):
        image_path=os.path.join(self.root_dir,self.annotation.iloc[index,0])
        cropped_image=Image.open(image_path).crop((self.annotation.iloc[index,1:5]))
        label=torch.tensor(label_to_ints[self.annotation.iloc[index,5]])
    
        #Resizes the cropped image to 224 x 224
        if self.transform:
            image=self.transform(cropped_image)
            return(image,label)

if __name__ == '__main__':

    #Read in training data
    train = pd.read_csv('./train_simple.csv')

    print("Training dataset")
    print(train)
    print(train.classname.value_counts())
    print("-----")

    if SHOW_IMAGES:
        visualize_training(random.choice(train.name.values))
        visualize_guess(random.choice(train.name.values), ['face_no_mask', 'face_with_mask', 'face_no_mask', 'face_with_mask', 'face_no_mask', 'face_with_mask'])

    #Training data makes up 75% of the dataset and testing data makes up the remaining 25%
    train_size=int(len(train)*0.75)
    test_size=len(train)-train_size
    image_transformer=transforms.Compose([transforms.Resize((224,224)),
                                 transforms.RandomCrop((224,224)),
                                 transforms.ToTensor()])

    image_path=os.path.join("./mask_data/dataset/masks/images")

    dataset = ImageStandardizer(train, image_path, image_transformer)

    batch_size=32

    trainset,testset = torch.utils.data.random_split(dataset,[train_size,test_size])
    training_data    = DataLoader(trainset, batch_size, True)
    testing_data     = DataLoader(testset, batch_size, True)

    dataiter=iter(training_data)
    images,labels=dataiter.next()

    if SHOW_IMAGES:
        for i in np.arange(20):
            fig,ax=plt.subplots(1)
            ax.axis('off')
            plt.imshow(np.transpose(images[i],(1,2,0)))
            plt.show()

    print(images.size())
    print(labels.size())

    #Download the pre-trained rsnet facial recognition model
    model = torchvision.models.resnet34(True)

    input_layer  = model.fc.in_features
    output_layer = nn.Linear(input_layer,2)
    model.fc.out_features = output_layer

    print(model.fc.out_features)

    #Sets the loss function
    criterion=nn.CrossEntropyLoss()

    #Sets the model learning rate
    optimizer=torch.optim.SGD(model.parameters(),lr=0.005)

    train_model(model, training_data)

    print(done)