import ast
import itertools
import logging
import pathlib
import zipfile
import six
import numpy
import json

import pandas
import jsonpickle

from . import agent, grid2d

PATHS_FILE_NAME = 'paths.csv'
AGENTS_FILE_NAME = 'agents.json'
GRID_FILE_NAME = 'grid.txt'
METADATA_FILE_NAME = 'metadata.json'

LOG = logging.getLogger(__name__)

def dump(target_path, agents=None, paths=None, grid = None, metadata: dict = None):
    if not agents and not paths:
        raise ValueError('At least one of agents, paths should be non-empty/none')

    with zipfile.ZipFile(target_path, 'w', zipfile.ZIP_DEFLATED) as z:
        if paths is None:
            try:
                paths_for_csv = [
                    [((s.x, s.y), s.step) for s in a.path]
                    for a in agents
                ]
            except Exception:
                paths_for_csv = []
        else:
            paths_for_csv = paths

        max_length = max(len(p) for p in paths_for_csv)
        for p in paths_for_csv:
            if len(p) < max_length:
                p.extend([None] * (max_length-len(p)))

        df = pandas.DataFrame(paths_for_csv)
        paths_csv_text = df.to_csv(None, index=False, header=False)

        z.writestr(PATHS_FILE_NAME, paths_csv_text)

        if agents is None:
            agent_dicts = None
        else:
            agent_dicts = [a.to_dict() for a in agents]

        for ad in agent_dicts:
            for p in ad.get('path', []):
                for attr in ['_agent', '_agent_repo', 'grid']:
                    try:
                        setattr(p, attr, None)
                    except (KeyError, AttributeError):
                        pass


        agents_dump = jsonpickle.dumps(agent_dicts)
        z.writestr(AGENTS_FILE_NAME, agents_dump)

        if grid:
            z.writestr(GRID_FILE_NAME, grid.to_str())

        if metadata:
            z.writestr(METADATA_FILE_NAME, json.dumps(metadata, indent=2))

class _LoadedPaths:
    def __init__(self, target_path: pathlib.Path):
        with zipfile.ZipFile(target_path, 'r') as z:
            paths_buff = z.read(PATHS_FILE_NAME).decode('utf-8')
            df = pandas.read_csv(six.StringIO(paths_buff), header=None)

            self.paths = []
            for _, r in df.iterrows():
                path = []
                for _, item in r.items():
                    if item is numpy.nan:
                        break

                    path.append( ast.literal_eval(item) )

                self.paths.append(path)

            agents_dump = z.read(AGENTS_FILE_NAME)
            agent_dicts = jsonpickle.loads(agents_dump)
            if agent_dicts:
                self.agents = [agent.Agent.from_dict(d) for d in agent_dicts]
            else:
                self.agents = None

            try:
                self.grid_raw = z.read(GRID_FILE_NAME).decode('utf-8')
                self.grid = grid2d.Grid2D.from_str(self.grid_raw)
            except Exception:
                LOG.warning(f'Failed loading grid from {target_path}', exc_info=True)
                self.grid_raw = None
                self.grid = None

            try:
                self.metadata = json.loads(z.read(METADATA_FILE_NAME))
            except Exception:
                self.metadata = None

    def get_makespan(self, ignore_adversarial=True):
        if self.agents:
            agents = self.agents
        else:
            agents = []

        return max( len(p) for a, p in itertools.zip_longest(agents, self.paths) if not a or not a.is_adversarial or not ignore_adversarial)

def load(target_path):
    return _LoadedPaths(target_path)
