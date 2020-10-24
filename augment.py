import pandas as pd

#Read in training data
df = pd.read_csv('train.csv')

#Grabs the data with the specified categories and writes to training file
mask_df = df.loc[(df['classname'] == "face_with_mask")]
no_mask_df = df.loc[(df['classname'] == "face_no_mask")]
#face_with_mask_incorrect = df.loc[(df['classname'] == "face_with_mask_incorrect")]
pd.concat([
    mask_df,
    no_mask_df]).to_csv('mask_no_mask.csv', index=False)