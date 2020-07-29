import pathlib
import pandas as pd
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askdirectory
import glob
import os

# plt.gca().set_color_cycle(['red', 'green', 'blue', 'yellow'])

Tk().withdraw()  # we don't want a full GUI, so keep the root window from appearing
path = askdirectory()  # show an "Open" dialog box and return the path to the selected file
all_files = glob.glob(path + "/*.csv")

paths = []
for root, dirs, files in os.walk(path):
    for file in files:
        if file.endswith(".csv") and ('03-mdr_on_normal_paths' in root):
            s = os.path.join(root, file)
            print(s)
            paths.append(s)

all_data = []

for filename in paths:
    file_data = []
    df = pd.read_csv(filename)
    file_data.append(df)
    frame = pd.concat(file_data, axis=0, ignore_index=True)
    df2 = frame.groupby(frame.adv_damage_steps)['mdr_run_time_seconds'].mean()
    df2.plot()
    all_data.append(df)

plt.legend(['Agents = 3', 'Agents = 2'], loc='upper left')
plt.ylabel('Time in Sec.')
plt.xlabel('Number of Damage Steps')
plt.show()

print(df2)