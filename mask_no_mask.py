import pandas as pd 
import os
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision import torch,datasets,transforms,models
from torch.utils.data import Dataset,DataLoader

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
        x1,x2,y1,y2=temp.values[i][2:6]
        #Draw the rectange around the face
        patch=patches.Rectangle((x1,x2),y1-x1,y2-x2,linewidth=2, 
                                edgecolor=class_color[temp.values[i][6:7][0]],facecolor="none",)
        #Add the text box
        ax.text(x1, x2, temp.values[i][6:7][0], style='italic',bbox={'facecolor': 'white', 'alpha': 0.75})
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
        x1,x2,y1,y2=temp.values[i][2:6]
        #Draw the rectange around the face
        patch=patches.Rectangle((x1,x2),y1-x1,y2-x2,linewidth=2, 
                                edgecolor=class_color[temp.values[i][6:7][0]],facecolor="none",)
        #Add the text box
        ax.text(x1, x2, labels[i], style='italic',bbox={'facecolor': 'white', 'alpha': 0.75})
        ax.add_patch(patch)

    #Plots the image with the label
    imgplot = plt.imshow(img)
    plt.show()

if __name__ == '__main__':

    #Read in training data
    train = pd.read_csv('./train_simple.csv')

    print("Training dataset")
    print(train)
    print(train.classname.value_counts())
    print("-----")

    visualize_training(random.choice(train.name.values))
    visualize_guess(random.choice(train.name.values), ['face_no_mask', 'face_with_mask', 'face_no_mask', 'face_with_mask', 'face_no_mask', 'face_with_mask'])