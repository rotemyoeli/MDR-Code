import pathlib
import pandas as pd
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askdirectory
import glob
import os


Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
path = askdirectory() # show an "Open" dialog box and return the path to the selected file
all_files = glob.glob(path + "/*.csv")


paths = []
for root, dirs, files in os.walk(path):
    for file in files:
        if file.endswith(".csv") and ('03-mdr_on_normal_paths' in root):
            print(root, dirs)
            print(os.path.join(root, file))
            s = os.path.join(root, file)
            print(s)
            paths.append(s)

all_data = []
for filename in paths:
    df = pd.read_csv(filename)
    all_data.append(df)


#li = []

#for filename in all_files:
#    df = pd.read_csv(filename, index_col=None, header=0)
#    li.append(df)

frame = pd.concat(all_data, axis=0, ignore_index=True)


#df = pd.read_csv(filename)

#df2 = df.groupby(df.paths_file_name.str[:21])['adv_agent_id'].count()
df2 = frame.groupby(frame.ms_mdr_path)['mdr_run_time_seconds'].mean()
df2.plot.line()

#df2.plot(kind='line', x='adv_agent_ds',y='mdr_online_const_run_time_seconds',color='green')
#plt.show()

print(df2)