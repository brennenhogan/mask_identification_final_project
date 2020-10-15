import pandas as pd 
import os
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision import torch,datasets,transforms,models
from torch.utils.data import Dataset,DataLoader
from PIL import Image
import numpy as np 

#Colors cooresponding to the labels
class_color={"face_no_mask":"r","face_with_mask":"g"}

# creating the function to visualize images and draw a box around the face
def visualize_training(image_name):
    #Reads in the image and creates the plot
    img=plt.imread(os.path.join('./mask_data/dataset/masks/images',image_name))
    temp=train[train.name==image_name]
    fig,ax=plt.subplots(1)
    ax.axis('off')
    fig.set_size_inches(20,10)
    ax.imshow(img)
    for i in range(len(temp)):
        #Grab the x and y bounds of the face from the training data
        x1,x2,y1,y2=temp.values[i][1:5]
        print(temp.values[i])
        #Draw the rectange around the face
        patch=patches.Rectangle((x1,x2),y1-x1,y2-x2,linewidth=2, 
                                edgecolor=class_color[temp.values[i][5]],facecolor="none",)
        #Add the text box
        ax.text(x1, x2, temp.values[i][5], style='italic',bbox={'facecolor': 'white', 'alpha': 0.75})
        ax.add_patch(patch)

    #Plots the image with the label
    imgplot = plt.imshow(img)
    plt.show()

# creating the function to visualize images and draw a box around the face
def visualize_guess(image_name, labels):
    #Reads in the image and creates the plot
    img=plt.imread(os.path.join('./mask_data/dataset/masks/images',image_name))
    temp=train[train.name==image_name]
    fig,ax=plt.subplots(1)
    ax.axis('off')
    fig.set_size_inches(20,10)
    ax.imshow(img)
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
    imgplot = plt.imshow(img)
    plt.show()

class MaskAndNoMask(Dataset): 
    def __init__(self,dataframe,root_dir,transform=None):
        self.annotation=dataframe
        self.root_dir=root_dir
        self.transform=transform
        
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self,index):
        print(self.annotation.iloc[index,0])
        print("T: {}".format(self.annotation.iloc[index,1:4]))
        img_path=os.path.join(self.root_dir,self.annotation.iloc[index,0])
        new_img=Image.open(img_path).crop((self.annotation.iloc[index,1:5]))
        label=torch.tensor(int(self.annotation.iloc[index,5]))
    
        if self.transform:
            image=self.transform(new_img)
            return(image,label)

if __name__ == '__main__':

    #Read in training data
    train = pd.read_csv('./train_simple.csv')

    print("Training dataset")
    print(train)
    print(train.classname.value_counts())
    print("-----")

    visualize_training(random.choice(train.name.values))
    visualize_guess(random.choice(train.name.values), ['face_no_mask', 'face_with_mask', 'face_no_mask', 'face_with_mask', 'face_no_mask', 'face_with_mask'])

    #Training data makes up 75% of the dataset and testing data makes up the remaining 25%
    train_size=int(len(train)*0.75)
    test_size=len(train)-train_size
    my_transform=transforms.Compose([transforms.Resize((224,224)),
                                 transforms.RandomCrop((224,224)),
                                 transforms.ToTensor()])

    path_images=os.path.join("./mask_data/dataset/masks/images")

    dataset=MaskAndNoMask(dataframe=train,root_dir=path_images,transform=my_transform)

    batch_size=32

    trainset,testset=torch.utils.data.random_split(dataset,[train_size,test_size])
    train_loader=DataLoader(dataset=trainset,batch_size=batch_size,shuffle=True)
    test_loader=DataLoader(dataset=testset,batch_size=batch_size,shuffle=True)

    dataiter=iter(train_loader)
    images,labels=dataiter.next()
    images=images.numpy()

    fig=plt.figure(figsize=(25,4))
    for idx in np.arange(20):
        ax=fig.add_subplot(2,20/2,idx+1,xticks=[],yticks=[])
        plt.imshow(np.transpose(images[idx],(1,2,0)))