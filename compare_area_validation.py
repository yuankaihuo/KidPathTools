import numpy as np
import pandas as pd

test_list = ['Case 03', 'Case 05', 'Case 09', 'Case 16']

csv_file = '/media/huoy1/MyDrive/pathology/data/kidpath_compareratio/ratios.csv'



df = pd.read_csv(csv_file)

df = df[df['subject'].isin(test_list)]

df = df[df['area_mask']>1000]

np.random.seed(0)
inds = np.random.permutation(df.shape[0])

circle_ratios = []
box_ratios = []
img_id = []
seg_id = []
subjects = []
for i in range(0,50):
    index = inds[i]
    circle_ratios.append(df.iloc[index]['circle_ratio'])
    box_ratios.append(df.iloc[index]['box_ratio'])
    img_id.append(df.iloc[index]['img_id'])
    seg_id.append(df.iloc[index]['seg_id'])
    subjects.append(df.iloc[index]['subject'])

circle_mean_ratio = np.array(circle_ratios).mean()
box_mean_ratio = np.array(box_ratios).mean()