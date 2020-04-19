from time import sleep

from pygame.locals import *
import pygame
import csv
import ast
import tkinter as tk
from tkinter import *
from tkinter import ttk as ttk, Entry
import os

from vgmapf.problems.mapf import paths_serializer


def get_x(route, step):
    if step < len(route):
        return route[step][0][0]
    return route[len(route)-1][0][0]


def get_y(route, step):
    if step < len(route):
        return route[step][0][1]
    return route[len(route)-1][0][1]

class Agent:
    agent_num = 0
    agent_route = []
    x = 0
    y = 0
    step = 0
    is_adversarial = False

    def __init__(self, num, route, is_adversarial=False):
        self.agent_num = num
        self.agent_route = route
        self.x = get_x(self.agent_route, 0)
        self.y = get_y(self.agent_route, 0)
        self.is_adversarial = is_adversarial
        self.zoom = 0

        #for ((x, y), step) in curr_route:

    def next_step(self):
        self.step += 1
        self.step = min(self.step, len(self.agent_route)-1)
        self.x = get_x(self.agent_route, self.step)
        self.y = get_y(self.agent_route, self.step)

    def prev_step(self):
        self.step -= 1
        self.step = max(self.step, 0)
        self.x = get_x(self.agent_route, self.step)
        self.y = get_y(self.agent_route, self.step)

    def get_start_x(self):
        return get_x(self.agent_route, 0)

    def get_start_y(self):
        return get_y(self.agent_route, 0)

    def get_goal_x(self):
        return get_x(self.agent_route, len(self.agent_route)-1)

    def get_goal_y(self):
        return get_y(self.agent_route, len(self.agent_route)-1)

    def restart(self):
        self.step = 0
        self.x = get_x(self.agent_route, 0)
        self.y = get_y(self.agent_route, 0)

class Maze:
    def __init__(self, map_file):
        with open(map_file, 'rt') as map_file:
            maze = [list(line.strip()) for line in map_file.readlines()]
            self.N = len(maze)
            self.M = len(maze[0])
            self.maze = maze

    def draw(self, display_surf, image_surf, zoom):
        shift_x = 0
        shift_y = 0

        for row_index in range(0, len(self.maze)):
            for col_index in range(0, len(self.maze[row_index])):
                if self.maze[row_index][col_index] == '@' or self.maze[row_index][col_index] == 'T':
                    display_surf.blit(image_surf, (col_index * zoom + shift_y, row_index * zoom + shift_x))


class PrintRoutesApp:
    windowWidth = 1024
    windowHeight = 860
    all_agents = []
    max_steps = 0
    time_step = 0
    zoom = 5
    step_label: Label
    root = None
    starting_row = 0
    stop_play = False
    pygame_frame = None
    scale = 1

    def __init__(self, tk_root, starting_row, zoom):
        self._running = True
        self._display_surf = None
        self._agent_img = None
        self._block_img = None
        self.root = tk_root
        self.starting_row = starting_row
        self.scale = 1
        self.zoom = zoom
        self.map_file = None
        self.routes_file = None

        start_col = 1
        top2 = Toplevel()
        button_frame = tk.Frame(top2, width=self.windowWidth, height=75)
        #button_frame.grid(row=self.starting_row, column=start_col+0)
        button_frame.grid(row=self.starting_row+1, column=0)
        button_frame.columnconfigure(0, weight=0)
        button_frame.rowconfigure(0, weight=0)

        step_title_label = Label(button_frame, text="Step:")
        step_title_label.grid(row=self.starting_row, column=start_col+0)
        self.step_label = Label(button_frame, text=self.time_step)
        self.step_label.grid(row=self.starting_row, column=start_col+1)
        prev_button = ttk.Button(button_frame, text='<', command=self.on_prev_button)
        prev_button.grid(row=self.starting_row, column=start_col+2)
        next_button = ttk.Button(button_frame, text='>', command=self.on_next_button)
        next_button.grid(row=self.starting_row, column=start_col+3)
        play_button = ttk.Button(button_frame, text='Play', command=self.on_play_button)
        play_button.grid(row=self.starting_row, column=start_col+4)
        play_button = ttk.Button(button_frame, text='Stop', command=self.on_stop_button)
        play_button.grid(row=self.starting_row, column=start_col+5)
        play_button = ttk.Button(button_frame, text='Restart', command=self.on_restart_button)
        play_button.grid(row=self.starting_row, column=start_col+6)

        next_button = ttk.Button(button_frame, text='-', command=self.on_zoomin_button)
        next_button.grid(row=self.starting_row, column=start_col + 7)
        next_button = ttk.Button(button_frame, text='+', command=self.on_zoomout_button)
        next_button.grid(row=self.starting_row, column=start_col + 8)

        # embed = tk.Frame(self.root, width=self.windowWidth, height=self.windowHeight)  # creates embed frame for pygame window
        self.pygame_frame = tk.Frame(self.root)  # creates embed frame for pygame window
        self.pygame_frame.grid(row=self.starting_row, column=0)
        self.pygame_frame.columnconfigure(0, weight=1)
        self.pygame_frame.rowconfigure(0, weight=1)

        os.environ['SDL_WINDOWID'] = str(self.pygame_frame.winfo_id())
        if sys.platform == 'win32':
            os.environ['SDL_VIDEODRIVER'] = 'windib'


    def print_routes(self, map_file, routes_file):
        self.time_step = 0
        self.map_file = map_file
        self.routes_file = routes_file

        self.maze = Maze(map_file)
        self.windowWidth = self.maze.M*self.zoom
        self.windowHeight = self.maze.N*self.zoom

        self.pygame_frame.configure(width=self.windowWidth, height=self.windowHeight)
        self.routes_file = routes_file
        self.create_agents_routes()

        self._display_surf = pygame.display.set_mode((self.windowWidth, self.windowHeight), HWSURFACE|DOUBLEBUF|RESIZABLE)
        pygame.display.update()
        self.root.update()
        self.on_execute()


    def create_agents_routes(self):
        self.all_agents = []

        lp = paths_serializer.load(self.routes_file)
        for curr_route, agnt in zip(lp.paths, lp.agents):
            self.max_steps = max(self.max_steps, len(curr_route))
            self.all_agents.append(Agent(agnt.id, curr_route, agnt.is_adversarial))

    def on_init(self):
        self._running = True
        images_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Images")
        self._agent_img = pygame.image.load(os.path.join(images_path, "small-drone.png")).convert()
        self._adversarial_agent_img = pygame.image.load(os.path.join(images_path, "small-drone1.png")).convert()
        self._block_img = pygame.image.load(os.path.join(images_path, "white-square-icon.png")).convert()
        self._start_img = pygame.image.load(os.path.join(images_path, "start.png")).convert()
        self._goal_img = pygame.image.load(os.path.join(images_path, "end.png")).convert()

    def on_event(self, event):
        if event.type == QUIT:
            self._running = False

    def on_loop(self):
        pass

    def on_render(self):
        if self._display_surf is None:
            return
        self._display_surf.fill((0, 0, 0))
        self.step_label.config(text=str(self.time_step))
        # first display all starts and goals
        for agent in self.all_agents:
            self._display_surf.blit(self._start_img, (self.zoom * agent.get_start_x() , self.zoom * agent.get_start_y()))
            self._display_surf.blit(self._goal_img, (self.zoom * agent.get_goal_x() , self.zoom * agent.get_goal_y()))

        for agent in self.all_agents:
            if agent.is_adversarial:
                self._display_surf.blit(self._adversarial_agent_img, (self.zoom * agent.x, self.zoom * agent.y))
            else:
                self._display_surf.blit(self._agent_img, (self.zoom * agent.x, self.zoom * agent.y))
        self.maze.draw(self._display_surf, self._block_img, self.zoom)

        self.root.update()
        pygame.display.flip()

    def on_cleanup(self):
        pygame.quit()

    def on_execute(self):
        self.on_init()

        while self._running:
            pygame.event.pump()
            keys = pygame.key.get_pressed()
            #event = pygame.event.wait()

            if keys[K_RIGHT]:
                self.on_next_button()

            if keys[K_LEFT]:
                self.on_prev_button()

            if keys[K_ESCAPE]:
                self._running = False

            ##########################
            ##########################
            if keys[K_MINUS]:
                self.scale /= 1.1
                #################################
                ###For zooming need to finish####
                #pygame.transform.rotozoom(self._display_surf, 0, self.scale)
                #################################

            elif keys[K_EQUALS]:
                self.scale *= 1.1


            #####################
            #####################

            self.on_render()
        self.on_cleanup()

    def on_next_button(self):
        self.time_step +=1
        self.time_step = min(self.time_step, self.max_steps)
        for agent in self.all_agents:
            agent.next_step()
        self.on_render()

    def on_prev_button(self):
        self.time_step -=1
        self.time_step = max(self.time_step, 0)
        for agent in self.all_agents:
            if self.time_step < len(agent.agent_route):
                agent.prev_step()
            else:
                agent.x = agent.get_goal_x()
                agent.y = agent.get_goal_y()

        self.on_render()

    def on_play_button(self):
        stop_play = False
        while self.time_step < self.max_steps and not stop_play:
            self.on_next_button()
            sleep(0.1)

    def on_stop_button(self):
        self.stop_play = True

    def on_restart_button(self):
        self.time_step = 0
        for agent in self.all_agents:
            agent.restart()
        self.on_render()

    def on_zoomin_button(self):
        self.zoom -= 1
        self.print_routes(self.map_file, self.routes_file)
        self.on_render()

    def on_zoomout_button(self):
        self.zoom += 1
        self.print_routes(self.map_file, self.routes_file)
        self.on_render()

# if __name__ == "__main__":
#     #temp hardcoded
#     map_file = 'D:/MAPF/Pathfinding Master 1.0/pathfinding/Data/Room1_new.txt'
#     routes_file = 'D:/MAPF/Pathfinding Master 1.0/pathfinding/Data/Robust_1/Route-1_Agents-3_Robust-1.csv'
#     theApp = App(tk.Tk(), 0)
#     theApp.print_routes(map_file, routes_file)
#
