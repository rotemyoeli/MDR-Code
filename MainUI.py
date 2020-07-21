#############################################################################
# import packages
##############################################################################
import math
import os
import tkinter as tk
from tkinter import *
from tkinter import ttk as ttk, Entry
from tkinter import filedialog
import main
import yaml
import ntpath
import pathlib


from pathfinding import SetupRoutes, config, MDR
from pathfinding.Core.Agent import Agent, MotionEquation, StartPolicy, GoalPolicy
from pathfinding.Utils import PrintRoutes
from pygame.locals import *
import pygame
from os.path import dirname, join

###################################################################################
# INPUT UI
###################################################################################
from pathfinding.Utils.PrintRoutesPyGame import PrintRoutesApp
import pathfinding
ROOT_DIR = pathlib.Path(pathfinding.__file__).parent.absolute()
project_root = os.path.dirname(os.path.dirname(__file__))
map_file_path = join(project_root, 'src/data')
output_file_path = join(project_root, 'src/outputs')

class MainUI:
    room_file_entry: Entry
    data_folder_entry: Entry
    robust_factor_spinbox: ttk.Spinbox
    num_of_agents_spinbox: ttk.Spinbox
    num_of_routes_spinbox: ttk.Spinbox
    num_of_routes_range_label: Label
    status_label: Label
    damage_steps_spinbox: Entry
    route_file_entry: Entry
    agentsEntries: []
    createRoutesFrame: ttk.LabelFrame
    include_step_num: IntVar
    print_routes_app: PrintRoutesApp

    ###################################################################################
    # Create UI
    ###################################################################################


    def main(self):
        main_dialog = self.create_main_dialog()

        # Create Tab Control
        TAB_CONTROL = ttk.Notebook(main_dialog)
        TAB_CONTROL.grid(row=0, sticky=NSEW)
        # Tab1
        create_routes_tab = ttk.Frame(TAB_CONTROL)
        TAB_CONTROL.add(create_routes_tab, text='Routes Creation')
        self.create_general_data_frame(create_routes_tab, 1)
        self.create_routes_frame(create_routes_tab, 2)

        # Tab2
        validate_routes_tab = ttk.Frame(TAB_CONTROL)
        TAB_CONTROL.add(validate_routes_tab, text='Validate Routes')

        # Tab3
        print_routes_tab = ttk.Frame(TAB_CONTROL)
        TAB_CONTROL.add(print_routes_tab, text='Print Routes')
        self.create_print_route_frame(print_routes_tab, 0)
        #top = Toplevel()
        #self.create_print_routes_pygame(top, 0)

        # Tab4
        pygame_routes_tab = ttk.Frame(TAB_CONTROL)
        TAB_CONTROL.add(pygame_routes_tab, text='Show Routes')
        #self.create_print_routes_pygame(pygame_routes_tab, 0)

        # Tab5
        mdr_tab = ttk.Frame(TAB_CONTROL)
        TAB_CONTROL.add(mdr_tab, text='MDR')
        self.create_MDR_frame(mdr_tab, 1)

        ttk.Separator(main_dialog, orient=HORIZONTAL).grid(row=11, sticky=EW, columnspan=10)
        self.create_run_buttons(main_dialog, 12)

        # Status Bar
        self.status_label = Label(main_dialog, text="Ready", bd=1, relief=SUNKEN, anchor=W)
        self.status_label.grid(row=100, column=0, columnspan=20, sticky=EW)

        main_dialog.grid_rowconfigure(10, weight=1)
        mainloop()

    def create_main_dialog(self):
        main_dialog = Tk()

        main_dialog.columnconfigure(0, weight=1)
        main_dialog.rowconfigure(0, weight=1)

        w = 590
        h = 670

        # get screen width and height
        ws = main_dialog.winfo_screenwidth()  # width of the screen
        hs = main_dialog.winfo_screenheight()  # height of the screen

        # calculate x and y coordinates for the Tk root window
        x = (ws / 2) - (w / 2)
        y = (hs / 2) - (h / 2)

        main_dialog.geometry('%dx%d+%d+%d' % (w, h, x, y))
        main_dialog.positionfrom()
        main_dialog.iconbitmap()  # TODO
        main_dialog.minsize(w, h)
        main_dialog.title("Multi Agent Path Finding 1.0")

        return main_dialog

    def create_general_data_frame(self, main_dialog, frame_row_index):
        frame = ttk.LabelFrame(main_dialog, text="General Data", padding=10)  # inside padding
        frame.grid(row=frame_row_index, pady=10, padx=10, sticky=NSEW, columnspan=10)  # outside padding
        frame.columnconfigure(0, weight=1)
        #frame.rowconfigure(0, weight=1)

        Label(frame, text="Room:").grid(row=0, column=1, sticky=W)
        self.room_file_entry = Entry(frame, width=65)
        self.room_file_entry.insert(0, map_file_path)

        self.room_file_entry.grid(row=0, column=2, sticky=W)
        ttk.Button(frame, text="Browse", command=self.browse_room_file).grid(row=0, column=3, sticky=E, padx=4)

        Label(frame, text="Data Folder:").grid(row=2, column=1, sticky=W)
        self.data_folder_entry = Entry(frame, width=65)
        self.data_folder_entry.insert(0, str(ROOT_DIR.joinpath(config.data_folder_default)))
        self.data_folder_entry.grid(row=2, column=2, sticky=W)
        ttk.Button(frame, text="Browse", command=self.browse_data_folder).grid(row=2, column=3, sticky=E, padx=4)

    def create_routes_frame(self, main_dialog, frame_row_index):
        self.createRoutesFrame = ttk.LabelFrame(main_dialog, text="Routes Data", padding=10)  # inside padding
        frame = self.createRoutesFrame
        frame.grid(row=frame_row_index, pady=10, padx=10, sticky=NSEW, columnspan=10)  # outside padding
        frame.columnconfigure(0, weight=1)
        #frame.rowconfigure(0, weight=1)

        Label(frame, text="Number of Agents:").grid(row=0, column=0, sticky=W, pady=3, padx=3, columnspan=2)
        self.num_of_agents_spinbox = ttk.Spinbox(frame, from_=1, to=30, width=8, command=self.num_of_agents_updated)
        self.num_of_agents_spinbox.grid(row=0, column=2, sticky=W)
        self.num_of_agents_spinbox.delete(0, 'end')
        self.num_of_agents_spinbox.insert(0, config.num_of_agents_default)

        Label(frame, text="Number of Routes:").grid(row=2, column=0, sticky=W, pady=3, padx=3, columnspan=2)
        self.num_of_routes_spinbox = ttk.Spinbox(frame, from_=1, to=30, width=8)
        self.num_of_routes_spinbox.grid(row=2, column=2, sticky=W)
        self.num_of_routes_spinbox.delete(0, 'end')
        self.num_of_routes_spinbox.insert(0, config.num_of_routes_default)
        maxRoutes = math.factorial(config.num_of_agents_default)
        self.num_of_routes_spinbox.configure(to=maxRoutes)
        self.num_of_routes_range_label = Label(frame, text="(Min:1 Max:" + str(maxRoutes) + ")")
        self.num_of_routes_range_label.grid(row=2, column=3, sticky=W, pady=3, padx=3, columnspan=2)

        # Label(frame, text="Robust Factor:").grid(row=1, column=0, sticky=W, pady=3, padx=3, columnspan=2)
        # self.robust_factor_spinbox = ttk.Spinbox(frame, from_=1, to=30, width=8)
        # self.robust_factor_spinbox.grid(row=1, column=2, sticky=W)
        # self.robust_factor_spinbox.delete(0, 'end')
        # self.robust_factor_spinbox.insert(0, config.robust_factor_default)

        # table header
        Label(frame, text="Must Reach\nTarget(%)").grid(row=4, column=1)
        Label(frame, text="Motion\nEquation").grid(row=4, column=2)
        Label(frame, text="Start Policy").grid(row=4, column=3)
        Label(frame, text="Goal Policy").grid(row=4, column=4)
        Label(frame, text="Adversarial").grid(row=4, column=5)
        Label(frame, text="D.S Budget").grid(row=4 , column=6)

        self.agentsEntries = []
        self.num_of_agents_updated()

    def create_print_route_frame(self, main_dialog, frame_row_index):

        frame = ttk.LabelFrame(main_dialog, text="Print Route", padding=10)  # inside padding
        frame.grid(row=0, pady=10, padx=10, sticky=NSEW, column=0)  # outside padding
        frame.columnconfigure(0, weight=1)


        #frame.rowconfigure(0, weight=1)

        Label(frame, text="Route:").grid(row=0, column=0, sticky=W, pady=3, padx=3)
        self.route_file_entry = Entry(frame, width=65)
        self.route_file_entry.insert(0, str(ROOT_DIR.joinpath(config.route_file_default)))
        self.route_file_entry.grid(row=0, column=1, sticky=W)
        ttk.Button(frame, text="Browse", command=self.browse_route_file).grid(row=0, column=2, sticky=E, padx=4)

        self.include_step_num = IntVar()
        Checkbutton(frame, text="Include Step No.", var=self.include_step_num).grid(row=1, column=0, columnspan=2, sticky=W)
        ttk.Button(frame, text="Print", command=self.print_routes).grid(row=1, column=2, sticky=E, padx=4)
        #ttk.Button(frame, text="Open New Window", command=self.open).grid(row=2, column=1, sticky=E, padx=4)

    def create_print_routes_pygame(self, main_dialog, frame_row_index):
        #top = Toplevel()
        frame = ttk.LabelFrame(main_dialog, padding=10)
        frame.grid(row=3, pady=10, padx=10, sticky=NSEW, column = 0)
        frame.columnconfigure(0, weight=1)
        #frame.rowconfigure(0, weight=1)
        self.print_routes_app = PrintRoutesApp(frame, 0, 5)

    def create_MDR_frame(self, main_dialog, frame_row_index):
        frame = ttk.LabelFrame(main_dialog, text="MDR Data", padding=10)  # inside padding
        frame.grid(row=frame_row_index, pady=10, padx=10, sticky=NSEW, columnspan=10)  # outside padding
        frame.columnconfigure(0, weight=1)
        #frame.rowconfigure(0, weight=1)

        Label(frame, text="Damage Steps Budget:").grid(row=0, column=1, sticky=W)
        self.damage_steps_spinbox = ttk.Spinbox(frame, from_=1, to=30, width=10)
        self.damage_steps_spinbox.grid(row=0, column=2, sticky=W)
        self.damage_steps_spinbox.delete(0, 'end')
        self.damage_steps_spinbox.insert(0, config.damage_steps_default)

    def create_run_buttons(self, main_dialog, row_index):
        ttk.Button(main_dialog, text='Create Routes',   command=self.create_routes)         .grid(row=row_index, column=0, sticky=W, pady=4, padx=5)
        #ttk.Button(main_dialog, text='Validate Routes',   command=self.not_implemented_yet) .grid(row=row_index, column=1, sticky=W, pady=4, padx=5)
        ttk.Button(main_dialog, text='Run MDR',         command=self.not_implemented_yet)   .grid(row=row_index+1, column=0, sticky=W, pady=4, padx=5)
        #top = Toplevel()
        #self.create_print_routes_pygame(main_dialog, row_index+1)

    ###################################################################################
    # Actions
    ###################################################################################
    #def open(self):
    #    top = Toplevel()
    #    top.title('Route View')
    #    btn2 = Button(top, text='Close Window',command=top.destroy).pack()


    def browse_room_file(self):
        self.room_file_entry.delete(0, 'end')
        self.room_file_entry.insert(0, filedialog.askopenfilename(initialdir=map_file_path))

    def browse_data_folder(self):
        self.data_folder_entry.delete(0, 'end')
        self.data_folder_entry.insert(0, filedialog.askdirectory(initialdir=output_file_path))

    def browse_route_file(self):
        self.route_file_entry.delete(0, 'end')
        self.route_file_entry.insert(0, filedialog.askopenfilename(initialdir=output_file_path))

    def create_routes(self):
        self.status_label.config(text='Running..')
        map_file_name = self.room_file_entry.get()
        data_folder = self.data_folder_entry.get()
        num_of_agents = int(self.num_of_agents_spinbox.get())
        num_of_routes = int(self.num_of_routes_spinbox.get())
        # get all agents data
        agents_data = []
        for agent_num in range(0, num_of_agents):
            agent_entries = self.agentsEntries[agent_num]
            # 1 Must reach target % - spinbox 0-100
            # 2 Motion Equation - combobox - 9/8/6/5
            # Start Policy - combobox - stay/ appear
            # Goal Policy - combobox - stay/ disappear
            # Adversarial - combobox - Yes/No
            # D.S Budget - combobox - NA/1/2/3...
            # print(motion_equation.current(), motion_equation.get())

            must_reach_target = int(agent_entries[1].get())
            motion_equation = MotionEquation(agent_entries[2].get())
            start_policy = StartPolicy(agent_entries[3].get())
            goal_policy = GoalPolicy(agent_entries[4].get())
            if agent_entries[5].get() == 'Yes':
                is_adversarial = True
                damage_steps_budget = agent_entries[6].get()
            else:
                is_adversarial = False
                damage_steps_budget = 0

            agents_data.append(Agent(agent_num, must_reach_target, start_policy, goal_policy, motion_equation,
                                     is_adversarial, int(damage_steps_budget)))

        # Update config-mdr file

        with open("config-mdr.yml", 'r') as stream:
            try:
                loaded = yaml.load(stream, Loader=yaml.FullLoader)
            except yaml.YAMLError as exc:
                print(exc)

        for item in loaded:
            print(item, loaded[item])
        print(len(agents_data), len(loaded['agents']))
        for x in range(0, len(agents_data)):
            loaded['agents'][x]['id'] = x+1
            loaded['agents'][x]['motion_equation'] = int(agents_data[x].motion_equation.value)
            if agents_data[x].is_adversarial==TRUE:
                loaded['agents'][x]['is_adversarial'] = True
                damage_stp = int(agents_data[x].damage_steps_budget)
                loaded['agents'][x]['damage_steps'] = damage_stp


        org_len = len(loaded['agents'])
        count = 0
        for y in range((len(loaded['agents']))-len(agents_data)):
            count += 1
            del loaded['agents'][org_len-count]

        # Add new record

        # Modify the fields from the dict
        head, tail = ntpath.split(map_file_name)
        loaded['map_file_name'] = join('data/', tail)
        loaded['permutations'] = num_of_routes
        loaded['robust_route'] = damage_stp
        #loaded['data_folder'] = data_folder
        #loaded['num_of_agents'] = num_of_agents

        # Save it again
        with open("config-mdr1.yml", 'w') as stream:
            try:
                yaml.dump(loaded, stream, default_flow_style=False)
            except yaml.YAMLError as exc:
                print(exc)

        config_file = join(pathlib.Path().absolute(), 'config-mdr1.yml')
        main.e2e_parallel(config_file)

        SetupRoutes.create_routes(map_file_name, data_folder, agents_data, num_of_routes)
        self.status_label.config(text='Ready')

    def num_of_agents_updated(self):
        newNumOfAgents = int(self.num_of_agents_spinbox.get())
        oldNumOfAgents = len(self.agentsEntries)

        # update maximum routes
        max_routes = math.factorial(newNumOfAgents)
        self.num_of_routes_spinbox.configure(to=max_routes)
        if int(self.num_of_routes_spinbox.get()) > max_routes:
            self.num_of_routes_spinbox.set(max_routes)
        self.num_of_routes_range_label.configure(text="(Min:1 Max:" + str(max_routes) + ")")

        if oldNumOfAgents < newNumOfAgents:
            # add new rows
            for i in range(oldNumOfAgents + 1, newNumOfAgents + 1):
                curr_row = i + 4
                cols = []

                # agent label
                agent_label = Label(self.createRoutesFrame, text="Agent " + str(i))
                agent_label.grid(row=curr_row, column=0, sticky=W)
                cols.append(agent_label)

                # Must reach target % - spinbox 0-100
                reach_target_spinbox = ttk.Spinbox(self.createRoutesFrame, from_=0, to=100, width=10)
                reach_target_spinbox.grid(row=curr_row, column=1, sticky=W)
                reach_target_spinbox.set(0)
                cols.append(reach_target_spinbox)

                # Motion Equation - combobox - 9/8/6/5
                motion_equation_values = [e.value for e in MotionEquation]
                motion_equation = ttk.Combobox(self.createRoutesFrame, values=motion_equation_values, width=10)
                motion_equation.grid(row=curr_row, column=2)
                motion_equation.current(3)
                cols.append(motion_equation)
                #print(motion_equation.current(), motion_equation.get())

                # Start Policy - combobox - stay/ appear
                start_policy_values = [e.value for e in StartPolicy]
                start_policy = ttk.Combobox(self.createRoutesFrame, values=start_policy_values, width=10)
                start_policy.grid(row=curr_row, column=3)
                start_policy.current(1)
                cols.append(start_policy)

                # Goal Policy - combobox - stay/ disappear
                goal_policy_values = [e.value for e in GoalPolicy]
                goal_policy = ttk.Combobox(self.createRoutesFrame, values=goal_policy_values, width=10)
                goal_policy.grid(row=curr_row, column=4)
                goal_policy.current(1)
                cols.append(goal_policy)

                # Adversarial - combobox - Yes/No
                is_adversarial = ttk.Combobox(self.createRoutesFrame, values=["Yes", "No"], width=10)
                is_adversarial.grid(row=curr_row, column=5)
                is_adversarial.current(1)
                if i == 1:
                    is_adversarial.current(0)
                cols.append(is_adversarial)

                # D.S Budget - combobox - NA/1/2/3... Or Spinbox disabled/enabled
                d_s_budget = ttk.Combobox(self.createRoutesFrame, values=["NA", "1", "2", "3", "4", "5"], width=10)
                d_s_budget.grid(row=curr_row, column=6)
                d_s_budget.current(0)
                if i == 1:
                    d_s_budget.current(3)
                cols.append(d_s_budget)
                self.agentsEntries.append(cols)

        else:
            #remove rows
            for i in range(oldNumOfAgents, newNumOfAgents, -1):
                for e in self.agentsEntries[i-1]:
                    e.grid_remove()
                del self.agentsEntries[i-1]

    def print_routes(self):
        top = Toplevel()

        self.create_print_routes_pygame(top, 3)
        map_file = self.room_file_entry.get()
        route_file = self.route_file_entry.get()
        self.print_routes_app.print_routes(map_file, route_file)


    def not_implemented_yet(self):
        print("Not Implemented Yet!")
        self.status_label.config(text='Running..')
        map_file_name = self.room_file_entry.get()
        data_folder = self.data_folder_entry.get()
        num_of_agents = int(self.num_of_agents_spinbox.get())
        num_of_routes = int(self.num_of_routes_spinbox.get())
        # get all agents data
        agents_data = []
        for agent_num in range(0, num_of_agents):
            agent_entries = self.agentsEntries[agent_num]
            # 1 Must reach target % - spinbox 0-100
            # 2 Motion Equation - combobox - 9/8/6/5
            # Start Policy - combobox - stay/ appear
            # Goal Policy - combobox - stay/ disappear
            # Adversarial - combobox - Yes/No
            # D.S Budget - combobox - NA/1/2/3...
            # print(motion_equation.current(), motion_equation.get())

            must_reach_target = int(agent_entries[1].get())
            motion_equation = MotionEquation(agent_entries[2].get())
            start_policy = StartPolicy(agent_entries[3].get())
            goal_policy = GoalPolicy(agent_entries[4].get())
            if agent_entries[5].get() == 'Yes':
                is_adversarial = True
                damage_steps_budget = agent_entries[6].get()
            else:
                is_adversarial = False
                damage_steps_budget = 0

            agents_data.append(Agent(agent_num, must_reach_target, start_policy, goal_policy, motion_equation,
                                     is_adversarial, int(damage_steps_budget)))

        MDR.simulator(map_file_name, data_folder, agents_data, num_of_routes)
        #SetupRoutes.create_routes(map_file_name, data_folder, agents_data, num_of_routes)
        self.status_label.config(text='Ready')




###################################################################################
# Main UI
###################################################################################
if __name__ == "__main__":
    mainUI = MainUI()
    mainUI.main()