import pickle
import pandas as pd
import numpy as np
with open('/Users/septem/Documents/dke/2/ke@work/second_phrase/data/cluster_result.pkl', 'rb') as f:
    cluster_result = pickle.load(f)
df = pd.read_csv('/Users/septem/Documents/dke/2/ke@work/code_for_ke@work/companies.csv')
df = df[['BEID','URL']]
df = df.set_index('BEID')

indices = []
# list only indices found in the cluster result
for key in cluster_result:
    if key != '':
        indices.append(int(key))
indices = np.array(indices)

df = df.loc[set(df.index).intersection(indices)]

df2 = pd.DataFrame(cluster_result, index=[0]).transpose()
df2 = df2.drop(df2[df2.index == ''].index)
df2.index = df2.index.astype(np.int32)
df = df.join(df2)
df = df.rename(columns={0:'label', 1:'prop'})
print(df.head())
print(df[df['label'] == 1])