import os
import pandas as pd
import pathlib
import tkinter as tk
from tkinter import filedialog
from tkinter.filedialog import askdirectory
import glob

os.chdir('..')
os.chdir('..')
os.chdir('outputs')
path1 = os.getcwd()

path = askdirectory(initialdir=path1,title='Select Folder') # shows dialog box and return the path


files_csv = glob.glob(path + "/*.csv")

df = pd.DataFrame()

for f in files_csv:
    df = pd.read_csv(f, header=0)
    #df = df.append(data)
mdr_ds_mean = df['ms_mdr_path'].mean()
select_rows = df.loc[df['ms_mdr_path'] >= mdr_ds_mean]

groupby_path = df.groupby(['paths_file_name']).mean()

groupby_path1 = groupby_path['ms_mdr_path'].nlargest(3)

array_paths = groupby_path1.to_numpy()

print(groupby_path1)


