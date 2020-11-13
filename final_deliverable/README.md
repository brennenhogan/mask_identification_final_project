# mask_identification_final_project
 AI & Social Good Final Project

 # Overview and Dataset

This project uses the mask labeled dataset from Kaggle. The dataset can be found and downloaded here: https://www.kaggle.com/wobotintelligence/face-mask-detection-dataset 

Please download the dataset and move it to the directory above where you are running the program before use
Call the data "mask_data" and it will be ignored by the .gitignore file for future use

Milestone 1 and 2 zip files have been added to the final directory. These zip files include the data from the milestone testing runs and the code necessary to collect the data.
No new data was collected for the final deliverable, as we focused on the stretch goal of the live demonstration and user interaction.

# Data paths
mask_data/dataset/masks/annotations
mask_data/dataset/masks/images

# Library and Running
To run please use python3
Necessary libraries: matplotlib, pytorch, pandas, numpy, ssl

# Data augmentation
python3 augment.py

# Main program
python3 3_class_custom.py
python3 3_class_resnet.py

After starting the program, you will be prompted to enter the mode. D begins demo mode and T begins testing mode.
The model also allows user to select a learning rate prior to starting the program. We have tested learning rates of 0.003 through 0.009.
Demo mode involves training the model on 75% of the data and making predictions based on user submitted inputs afterwards.
Test mode involves training the model on 75% of the data and testing predictions on the final 25% of the data.

