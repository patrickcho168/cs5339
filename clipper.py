import pandas as pd
import os

clip = 0.90
classes = 8

def clip_csv(csv_file, clip, classes):
    # Read the submission file
    df = pd.read_csv(csv_file, index_col=0)

    # Clip the values
    df = df.clip(lower=(1.0 - clip)/float(classes - 1), upper=clip)
    
    # Normalize the values to 1
    df = df.div(df.sum(axis=1), axis=0)

    # Save the new clipped values
    df.to_csv('clip.csv')
    print(df.head(10))
    
# Of course you are going to use your own submission here
clip_csv('./sub-test2-64.csv', clip, classes)
