'''
May 2017
@author: Burkhard A. Meier
'''
#======================
# imports
#======================
import tkinter as tk
from tkinter import ttk
import pathlib
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
from tkinter.filedialog import askdirectory
import glob
import os
from pathlib import Path

full_path = askdirectory()  # show an "Open" dialog box and return the path to the selected file
dir_name = os.path.basename(full_path)
parent_path = str(Path(full_path).parents[1])

all_files = glob.glob(parent_path + "/*.csv")

paths = []
for root, dirs, files in os.walk(parent_path):
    for file in files:
        if file.endswith(".csv") and (dir_name in root):
            s = os.path.join(root, file)
            paths.append(s)

# Get sample data structure
data_structure = []
for filename in paths:
    file_data = []
    data_structure = pd.read_csv(filename)

# Create instance
win = tk.Tk()

# Add a title
win.title("Python GUI")

# Modify adding a Label
a_label = ttk.Label(win, text="A Label")
a_label.grid(column=0, row=0)

# Modified Button Click Function
def click_me():
    legend_list = []
    for filename in paths:
        file_data = []
        legend_path = str((os.path.basename(os.path.dirname(str(Path(filename).parents[0])))) + ' Agents')
        legend_list.append(legend_path)
        df = pd.read_csv(filename)
        file_data.append(df)
        frame = pd.concat(file_data, axis=0, ignore_index=True)
        if operation_chosen.get() == 'mean':
            df2 = frame.groupby(field_x_chosen.get())[field_y_chosen.get()].mean()
        else:
            df2 = frame.groupby(field_x_chosen.get())[field_y_chosen.get()].sum()
        df2.plot()
        #all_data.append(df)
    plt.legend(legend_list, loc='upper left')
    plt.ylabel(field_y_chosen.get())
    plt.xlabel(field_x_chosen.get())
    plt.show()


# Adding a Button
action = ttk.Button(win, text="Run Report!", command=click_me)
action.grid(column=3, row=1)                                 # <= change column to 2

# Adding X var combobox
ttk.Label(win, text="Choose X Var, GroupBy:").grid(column=0, row=0)
field_x = tk.StringVar()
field_x_chosen = ttk.Combobox(win, width=25, textvariable=field_x, state='readonly')
com_val = []
for combo_val in range(0, len(data_structure.columns)):
    com_val.append(data_structure.columns[combo_val])
field_x_chosen['values'] = com_val
field_x_chosen.grid(column=0, row=1)
field_x_chosen.current(0)


# Adding Y var combobox
ttk.Label(win, text="Choose Y Var:").grid(column=1, row=0)
field_y = tk.StringVar()
field_y_chosen = ttk.Combobox(win, width=25, textvariable=field_y, state='readonly')
com_val = []
for combo_val in range(0, len(data_structure.columns)):
    com_val.append(data_structure.columns[combo_val])
field_y_chosen['values'] = com_val
field_y_chosen.grid(column=1, row=1)
field_y_chosen.current(0)

# Adding choose operation combobox
ttk.Label(win, text="Choose Operation:").grid(column=2, row=0)
operation = tk.StringVar()
operation_chosen = ttk.Combobox(win, width=25, textvariable=operation, state='readonly')
operation_chosen['values'] = ('mean', 'sum')
operation_chosen.grid(column=2, row=1)
operation_chosen.current(0)

#field_chosen.focus()      # Place cursor into name Entry
#======================
# Start GUI
#======================
win.mainloop()