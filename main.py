import inspect
import itertools
import multiprocessing
import os
import pprint
import sys
import pathlib
import timeit
import datetime
from os import listdir
from os.path import isfile, join

import logging.config
import random
from typing import Any, Tuple

import argh
import six
import csv

from dataclasses import dataclass

from vgmapf.problems.mapf.mdr_finder import RobustPathMode

from vgmapf.search.base_algorithm import SearchObserver, NodeCounter
from vgmapf.problems import sliding_tile_puzzle
from vgmapf.problems.mapf import pathfinding, agent, mdr_finder, cbs, agent_repository, kamikaze_planner
from vgmapf.problems.mapf import heuristics as mapf_heuristics
from vgmapf.search.algorithms import astar, phs, ucs, bfs, dfs
from vgmapf.utils import logutils, benchmark_utils
from vgmapf.utils import time_utils
from vgmapf.problems.mapf import grid2d
from vgmapf.problems.mapf import multi_agent_pathfinding
from vgmapf import config
from vgmapf.problems.mapf import paths_serializer

STAGE_04_MDR_ROBUST_PATHS = '04-mdr_on_robust_paths'
STAGE_03_MDR_NORMAL_PATHS = '03-mdr_on_normal_paths'
STAGE_025_KAMIKAZE = '025-kamikaze_on_robust_paths'
STAGE_02_ROBUST_PATHS = '02-robust_paths'
STAGE_01_NORMAL_PATHS = '01-normal_paths'

LOG = logging.getLogger(__name__)


def test_tile_puzzle():
    init_state = sliding_tile_puzzle.State.random(4, 4, 20)
    init_state.dump()

    solvers = [
        (phs.Searcher, dict(h_func=sliding_tile_puzzle.h_manhatten_distance)),
        (astar.Searcher, dict(h_func=sliding_tile_puzzle.h_manhatten_distance)),
    ]

    results = []
    for klass, kwargs in solvers:
        counter = NodeCounter()
        searcher = klass(init_state.clone(), counter, **kwargs)
        path, cost = searcher.search()
        results.append((klass, cost, counter.count))

    for klass, cost, expanded_nodes in results:
        print(f'{klass.__name__}: solution cost={cost}, nodes={expanded_nodes}')


def test_grid(map_file_name: str):
    map_path = pathlib.Path(map_file_name)
    g = grid2d.Grid2D.from_file(map_path)
    print(g.to_str())
    from IPython import embed;
    embed()


def test_pathfinding(map_file_name: str, start='6,4', end='33,31'):
    start = tuple([int(x) for x in start.split(',')])
    end = tuple([int(x) for x in end.split(',')])

    g = grid2d.Grid2D.from_file(pathlib.Path(map_file_name))
    start_state = pathfinding.PathfindingState(g, start, 0, start, end)

    print(g.to_str(start=start_state.cell, end=start_state.goal_cell))

    h_func = mapf_heuristics.manhatten

    solvers = [
        (phs.Searcher, dict(h_func=h_func)),
        (astar.Searcher, dict(h_func=h_func)),
        (ucs.Searcher, {}),
        (bfs.Searcher, {}),
        # (dfs.Searcher, {}),
    ]

    results = []
    for klass, kwargs in solvers:
        counter = NodeCounter()
        searcher = klass(start_state.clone(), counter, **kwargs)
        LOG.debug(f'START: {searcher}')
        print('\n')
        try:
            path, cost = searcher.search()  # type: Tuple[List[pathfinding.PathfindingState], int]
        except Exception:
            path = None
            cost = 0
        LOG.debug(f'FINISH: {searcher}')
        results.append((searcher, cost, counter.count))

    for searcher, cost, expanded_nodes in results:
        LOG.info(f'{searcher.__class__.__name__}: solution cost={cost}, length:{len(path)}, nodes={expanded_nodes}')
        print(g.to_str(path=[x.cell for x in path], start=start_state.cell, end=start_state.goal_cell))


@argh.arg('-p', '--permutations', type=int, default=None)
def test_cbs(run_config_file_name: str, out_file_name: str = None, random_seed=None, permutations=None):
    rc = config.load(run_config_file_name)

    if permutations is not None:
        rc.permutations = permutations

    random.seed(random_seed)

    if out_file_name:
        base_out_path = pathlib.Path(out_file_name)
    else:
        base_out_path = pathlib.Path(rc.map_file_name).parent.joinpath(
            f'routes-{time_utils.get_current_time_stamp()}.csv')

    g = grid2d.Grid2D.from_file(pathlib.Path(rc.map_file_name))

    if not rc.start:
        rc.start = g.get_random_free_cell()

    if not rc.end:
        rc.end = g.get_random_free_cell({rc.start})

    agent_count = len(rc.agents)

    agents_have_start = False
    agents_have_end = False
    for a in rc.agents:
        if a.get('start_cell'):
            agents_have_start = True
        if a.get('goal_cell'):
            agents_have_end = True

    if not agents_have_start:
        start_cells = [rc.start] + g.find_free_cells_around(rc.start, agent_count - 1)
    else:
        start_cells = [a['start_cell'] for a in rc.agents]

    if not agents_have_end:
        end_cells = [rc.end] + g.find_free_cells_around(rc.end, agent_count - 1, set(start_cells))
    else:
        end_cells = [a['goal_cell'] for a in rc.agents]

    for a, sc, gc in zip(rc.agents, start_cells, end_cells):
        a['start_cell'] = sc
        a['goal_cell'] = gc

    LOG.info(f'STARTING mapf test, run_config: {rc}, base_out_path: {base_out_path}')

    for permutation_idx in range(rc.permutations):
        LOG.info(f'STARTED permutation {permutation_idx:03d}/{rc.permutations:03d}')
        # random.shuffle(rc.agents)
        agents = [agent.Agent(**a) for a in rc.agents]
        cbs_finder = cbs.CbsMafpFinder(g)
        agents_repo, total_cost = cbs_finder.find_path(agent_repository.AgentRepository(agents), astar.Searcher,
                                                       lambda agnt: dict(
                                                           h_func=mapf_heuristics.get_good_manhatten_like_heuristic(
                                                               agnt)))

        for a in agents_repo.agents:
            LOG.debug(
                f"[{permutation_idx:03d}/{rc.permutations:3d}]:: Agent: {a.id}, path len: {len(a.path)} path cost: "
                f"{a.path_cost}, expanded nodes: {a.expanded_nodes}")
            print(g.to_str(a.cells_path(), a.start_cell, a.goal_cell, path_chr=str(a.id)[0]))

        out_path = base_out_path.parent / (base_out_path.stem + f'-{permutation_idx:03d}' + base_out_path.suffix)

        cbs_finder.save_paths(agents_repo, out_path)
        LOG.info(f'FINISHED permutation {permutation_idx:03d}/{rc.permutations:03d} => {out_path}')
        cbs_finder.validate_paths(g, agents_repo)


@argh.arg('-p', '--permutations', type=int, default=None)
def mapf(run_config_file_name: str, out_file_name: str = None, random_seed=None, permutations: int = None,
         map_file_name: str = None):
    rc = config.load(run_config_file_name)

    if permutations is not None:
        rc.permutations = permutations

    if map_file_name:
        rc.map_file_name = map_file_name

    random.seed(random_seed)

    if out_file_name:
        base_out_path = pathlib.Path(out_file_name)
    else:
        timestamp = time_utils.get_current_time_stamp()
        base_out_path = pathlib.Path(__file__).parent.joinpath(
            'routes',
            timestamp,
            f'paths-{timestamp}.path'
        )
        base_out_path.parent.mkdir(parents=True, exist_ok=True)

    g = grid2d.Grid2D.from_file(pathlib.Path(rc.map_file_name))

    _update_start_and_goal_cells(rc, g)

    LOG.info(f'STARTING mapf test, run_config: {rc}, base_out_path: {base_out_path}')

    for permutation_idx in range(rc.permutations):
        with benchmark_utils.time_it(f'Building path #{permutation_idx}'):
            LOG.info(f'STARTED permutation {permutation_idx + 1:03d}/{rc.permutations:03d}')
            if permutation_idx > 0:
                random.shuffle(rc.agents)
            agents = [agent.Agent(**a) for a in rc.agents]
            mf = multi_agent_pathfinding.MapfFinder(g, agents)
            mf.find_paths(astar.Searcher,
                          lambda agnt: dict(h_func=mapf_heuristics.get_good_manhatten_like_heuristic(agnt)))

            for a in mf.agents:
                LOG.debug(
                    f"[{permutation_idx + 1:03d}/{rc.permutations:03d}]:: Agent: {a.id}, path len: {len(a.path)} "
                    f"path cost: {a.path_cost}, expanded nodes: {a.expanded_nodes}")

                print(g.to_str(a.cells_path(), a.start_cell, a.goal_cell, path_chr=str(a.id)[0]))

            out_path_base = base_out_path.parent / (
                    base_out_path.stem + f'-{permutation_idx:03d}' + base_out_path.suffix)
            mf.save_paths(out_path_base)
            LOG.info(f'FINISHED permutation {permutation_idx + 1:03d}/{rc.permutations:03d} => {out_path_base}')
            mf.validate_paths()

            robust_route = RobustPathMode(rc.robust_route)

            if robust_route == RobustPathMode.OFFLINE:
                makespan_original = mf.agents_repo.get_makespan()

                for agnt in agents:
                    if not agnt.is_adversarial:
                        agnt.path = None
                        agnt.path_cost = 0
                        agnt.expanded_nodes = 0

                mf_robust = multi_agent_pathfinding.MapfFinder(g, agents,
                                                               adv_agent_radiuses={a.id: a.damage_steps * 2 for a in
                                                                                   agents
                                                                                   if a.is_adversarial})
                mf_robust.find_paths(astar.Searcher,
                                     lambda agnt: dict(h_func=mapf_heuristics.get_good_manhatten_like_heuristic(agnt)))

                for a in mf_robust.agents:
                    LOG.debug(
                        f"[{permutation_idx + 1:03d}/{rc.permutations:03d}]:: Agent: {a.id}, path len: {len(a.path)} "
                        f"path cost: {a.path_cost}, expanded nodes: {a.expanded_nodes}")

                    print(g.to_str(a.cells_path(), a.start_cell, a.goal_cell, path_chr=str(a.id)[0]))

                out_path_robust = base_out_path.parent / (base_out_path.stem + f'-{permutation_idx:03d}-robust'
                                                          + base_out_path.suffix)
                mf_robust.save_paths(out_path_robust)
                LOG.info(f'FINISHED permutation {permutation_idx + 1:03d}/{rc.permutations:03d} => {out_path_robust}')
                mf_robust.validate_paths()
                makespan_robust = mf_robust.agents_repo.get_makespan()

                LOG.info(f'The difference in makespan is {makespan_robust - makespan_original}')

    return base_out_path.parent


def _get_start_and_end_base_cells(rc, g):
    start = rc.start
    end = rc.end
    if not start:
        start = g.get_random_free_cell()
    if not end:
        end = g.get_random_free_cell({rc.start})

    return start, end

def _update_start_and_goal_cells(rc: config.RunConfig, g: grid2d.Grid2D):
    if not rc.start or not rc.end:
        is_random = True
    else:
        is_random = False

    while True:
        start, end = _get_start_and_end_base_cells(rc, g)
        if not is_random:
            rc.start, rc.end = start, end
            break

        distance = mapf_heuristics.manhattan_distance(abs(start[0] - end[0]), abs(start[1] - end[1]))
        if rc.min_start_end_distance <= distance <= rc.max_start_end_distance:
            LOG.debug( f'Chose start={rc.start}, end={rc.end}. distance={distance:.2f}. moving on')
            rc.start, rc.end = start, end
            break

    agent_count = len(rc.agents)
    agents_have_start = False
    agents_have_end = False
    for a in rc.agents:
        if a.get('start_cell'):
            agents_have_start = True
        if a.get('goal_cell'):
            agents_have_end = True

    if agent_count > 6:
        radius = 2
    else:
        radius = 1

    if not agents_have_start:
        # start_cells = [rc.start] + g.find_free_cells_in_row(rc.start, agent_count - 1)
        start_cells = [rc.start] + g.find_free_cells_around(rc.start, agent_count - 1, radius=radius)
        random.shuffle(start_cells)
    else:
        start_cells = [a['start_cell'] for a in rc.agents]
    if not agents_have_end:
        end_cells = [rc.end] + g.find_free_cells_around(rc.end, agent_count - 1, set(start_cells), radius=radius)
    else:
        end_cells = [a['goal_cell'] for a in rc.agents]
    for a, sc, gc in zip(rc.agents, start_cells, end_cells):
        a['start_cell'] = sc
        a['goal_cell'] = gc


@argh.arg('adv_agent_id', type=int)
@argh.arg('adv_agent_ds', type=int)
def mdr(paths_file: str, adv_agent_id: int, adv_agent_ds: int, out_file_name: str = None, robust_mode: str = 'DISABLE'):
    logging.getLogger('vgmapf.problems.mapf.multi_agent_pathfinding').setLevel(logging.INFO)
    paths_file = pathlib.Path(paths_file)
    lp = paths_serializer.load(paths_file)
    adv_agent = [a for a in lp.agents if a.id == adv_agent_id][0]
    adv_agent.is_adversarial = True
    adv_agent.damage_steps = adv_agent_ds

    robust_mode = getattr(mdr_finder.RobustPathMode, robust_mode)

    if not out_file_name:
        out_file_name = paths_file.parent / (
                paths_file.stem + f'-mdr-a_{adv_agent_id}-ds_{adv_agent_ds}' + paths_file.suffix)

    paths = {a.id: a.path for a in lp.agents}
    original_makespan = mdr_finder.get_makespan(paths, lp.agents)

    mdrf = mdr_finder.MaxDamageRouteFinder(lp.grid, lp.agents, astar.Searcher, lambda agnt: dict(
        h_func=mapf_heuristics.get_good_manhatten_like_heuristic(agnt)), robust_mode)
    goal_state, info = mdrf.find()
    paths = goal_state.paths

    for aid, p in paths.items():
        p_cells = [x.cell for x in p]
        LOG.info(f'Agent [{aid:02d}], path length: {len(p)}')
        print(lp.grid.to_str(p_cells, p_cells[0], p_cells[-1], path_chr=str(aid)[0]))

    mdr_makespan = mdr_finder.get_makespan(paths, lp.agents)
    paths_serializer.dump(out_file_name, lp.agents, [paths[a.id] for a in lp.agents], lp.grid)

    LOG.info(f'Original makespan: {original_makespan} | MDR makespan: {mdr_makespan} | MDR info: {info}')

    return info, original_makespan, mdr_makespan

@argh.arg('adv_agent_id', type=int)
@argh.arg('adv_agent_ds', type=int)
def kamikaze(paths_file: str, adv_agent_id: int, adv_agent_ds: int, out_file_name: str = None, robust_mode: str = 'DISABLE'):
    logging.getLogger('vgmapf.problems.mapf.multi_agent_pathfinding').setLevel(logging.INFO)
    paths_file = pathlib.Path(paths_file)
    lp = paths_serializer.load(paths_file)
    adv_agent = [a for a in lp.agents if a.id == adv_agent_id][0]
    adv_agent.is_adversarial = True
    adv_agent.damage_steps = adv_agent_ds

    robust_mode = getattr(mdr_finder.RobustPathMode, robust_mode)

    if not out_file_name:
        out_file_name = paths_file.parent / (
                paths_file.stem + f'-kamikaze-a_{adv_agent_id}-ds_{adv_agent_ds}' + paths_file.suffix)

    paths = {a.id: a.path for a in lp.agents}
    original_makespan = mdr_finder.get_makespan(paths, lp.agents)

    kp = kamikaze_planner.KamikazePlanner(lp.grid, lp.agents, astar.Searcher, lambda agnt: dict(
        h_func=mapf_heuristics.get_good_manhatten_like_heuristic(agnt)), robust_mode,
    )
    goal_state, info = kp.find()
    paths = goal_state.paths
    assert isinstance(goal_state, kamikaze_planner.KamikazeState)
    collision = goal_state.get_collision()

    for aid, p in paths.items():
        p_cells = [x.cell for x in p]
        LOG.info(f'Agent [{aid:02d}], path length: {len(p)}')
        print(lp.grid.to_str(p_cells, p_cells[0], p_cells[-1], path_chr=str(aid)[0]))

    mdr_makespan = mdr_finder.get_makespan(paths, lp.agents)
    paths_serializer.dump(out_file_name, lp.agents, [paths[a.id] for a in lp.agents], lp.grid)

    LOG.info(f'Collision: {collision} | Original makespan: {original_makespan} | MDR makespan: {mdr_makespan} | MDR info: {info}')

    return info, original_makespan, mdr_makespan

def load(paths_file):
    lp = paths_serializer.load(paths_file)
    from IPython import embed;
    embed()

def _extract_result_row(path_file: pathlib.Path):
    lp = paths_serializer.load(path_file)
    row = lp.metadata.copy()
    agents_count = len(lp.agents)
    row['agents_count'] = agents_count
    return row

def _normalize_row_fields(rows):
    """
    Makes sure that all rows have the same set of keys
    """

    exclude_fields = ['agents_metadata', 'agents', 'map_file_name','paths_file_name']

    all_keys = set()
    for r in rows:
        for k in r.keys():
            all_keys.add(k)

    for r in rows:
        for k in all_keys:
            if k not in r:
                r[k] = ''

        for k in exclude_fields:
            try:
                del r[k]
            except KeyError:
                pass

        adv_radiuses = r.get('adv_radiuses', None)
        if adv_radiuses:
            assert len(adv_radiuses) == 1
            del r['adv_radiuses']
            r['robust_radius'] = next(iter(adv_radiuses.values()))


_HEADER_FIELDS_ORDER = dict(
    map_name = '__01',
    experiment_name = '__02',
    path_name = '__03',
    agents_count = '__04',
    robust_radius = '__05',
    comment = 'zzzzzz_01'
)

def _get_header_fields(rows):
    fields = rows[0].keys()
    fields = sorted(fields, key = lambda x: _HEADER_FIELDS_ORDER.get(x, x))
    return fields

def gather_results(root_dir: str):
    root_path = pathlib.Path(root_dir)
    out_dir_path = root_path

    for stage_dir_name in [STAGE_01_NORMAL_PATHS, STAGE_02_ROBUST_PATHS, STAGE_025_KAMIKAZE, STAGE_03_MDR_NORMAL_PATHS,
                           STAGE_04_MDR_ROBUST_PATHS]:
            out_file_path = out_dir_path / f'{stage_dir_name}.csv'
            rows = []
            with logutils.log_context(f'stage: {stage_dir_name}'):
                for map_results_path in root_path.iterdir():
                    if not map_results_path.is_dir():
                        continue
                    map_name = map_results_path.name
                    with logutils.log_context(f'\tmap_name: {map_name}'):
                        for experiment_path in map_results_path.iterdir():
                            experiment_name = experiment_path.name
                            with logutils.log_context(f'\t\texperiment: {experiment_name}'):
                                stage_path = experiment_path / stage_dir_name
                                for path_file in stage_path.glob('*.path'):
                                    # with logutils.log_context(f'\t\t\tpath_file: {path_file}'):
                                    r = _extract_result_row(path_file)
                                    r.update(path_name = path_file.stem, map_name = map_name, experiment_name = experiment_name)
                                    rows.append(r)

            _normalize_row_fields(rows)
            header_fields = _get_header_fields(rows)
            with out_file_path.open('w', newline='') as fout:
                out_csv = csv.DictWriter(fout, fieldnames=header_fields)
                out_csv.writeheader()
                [out_csv.writerow(r) for r in rows]

def _stage_1_normal_paths(normal_paths_dir, grid, rc, permutation_idx):
    try:
        with benchmark_utils.time_it(f'Building path #{permutation_idx}'):
            LOG.info(f'STARTED permutation {permutation_idx + 1:03d}/{rc.permutations:03d}')
            if permutation_idx > 0:
                random.shuffle(rc.agents)
            agents = [agent.Agent(**a) for a in rc.agents]

            with benchmark_utils.time_it() as t:
                mf = multi_agent_pathfinding.MapfFinder(grid, agents)
                mf.find_paths(astar.Searcher,
                              lambda agnt: dict(h_func=mapf_heuristics.get_good_manhatten_like_heuristic(agnt)))
            mf.validate_paths()

            out_path = normal_paths_dir / f'{permutation_idx:03d}.path'
            mf.save_paths(out_path, metadata=dict(
                mapf_run_time_sec=t.getElapsed(),
                makespan=mf.agents_repo.get_makespan(only_non_adversarial=False),
                agents_metadata=[
                    dict(
                        id=a.id,
                        start_cell=a.start_cell,
                        goal_cell=a.goal_cell,
                        path_cost=a.path_cost,
                        path_expanded_nodes=a.expanded_nodes,
                        motion_equation=a.motion_equation.name,
                        start_policy=a.start_policy.name,
                        goal_polciy=a.goal_policy.name,
                    )
                    for a in agents
                ]
            ))
            LOG.info(f'FINISHED permutation {permutation_idx + 1:03d}/{rc.permutations:03d} => {out_path}')
    except Exception as e:
        LOG.error(e, exc_info=True)

def _stage_2_normal_robust(robust_paths_dir, grid, max_adv_agent_ds, p):
    try:
        LOG.info(f'STARTED normal robust on {p}')
        lp = paths_serializer.load(p)
        for org_adv_agent in lp.agents:
            adv_agent_id = org_adv_agent.id
            agents = [a.clone(clear_path=False) for a in lp.agents]

            adv_agent = [a for a in agents if a.id == adv_agent_id][0]

            LOG.info(f'STARTED Robust paths with agent {adv_agent.id}')
            for adv_agent_ds in range(1, max_adv_agent_ds + 1):
                for robust_radius in range(1, 2*adv_agent_ds+1):
                    LOG.info(f'STARTED Robust paths with agent {adv_agent.id} and DS={adv_agent_ds} '
                             f'and Robust Radius={robust_radius}')
                    adv_agent.is_adversarial = True
                    adv_agent.damage_steps = adv_agent_ds

                    for a in agents:
                        if a is not adv_agent:
                            a.is_adversarial = False
                            a.damage_steps = 0
                            a.path = None
                    try:
                        with benchmark_utils.time_it() as t:
                            mf = multi_agent_pathfinding.MapfFinder(grid, agents,
                                                                    adv_agent_radiuses={adv_agent.id: robust_radius})
                            mf.find_paths(astar.Searcher,
                                          lambda agnt: dict(
                                              h_func=mapf_heuristics.get_good_manhatten_like_heuristic(agnt)))
                        mf.validate_paths()

                        out_path = robust_paths_dir / f'robust_path-{p.stem}-agent_{adv_agent.id}-ds_{adv_agent_ds}' \
                                                      f'-rr_{robust_radius}{p.suffix}'
                        mf.save_paths(out_path, metadata=dict(
                            mapf_run_time_sec=t.getElapsed(),
                            makespan=mf.agents_repo.get_makespan(only_non_adversarial=False),
                            adv_radiuses=mf.adv_agent_radiuses,
                            agents=[
                                dict(
                                    id=a.id,
                                    start_cell=a.start_cell,
                                    goal_cell=a.goal_cell,
                                    path_cost=a.path_cost,
                                    path_expanded_nodes=a.expanded_nodes,
                                    motion_equation=a.motion_equation.name,
                                    start_policy=a.start_policy.name,
                                    goal_polciy=a.goal_policy.name,
                                    is_adversarial=a.is_adversarial,
                                    damage_steps=a.damage_steps,
                                )
                                for a in agents
                            ]
                        ))
                    except Exception as e:
                        LOG.error(
                            f'Failed creating robust route for {org_adv_agent}, ds={adv_agent_ds}, rr={robust_radius}:'
                            f' {e}, moving on...')

        LOG.info(f'FINISHED normal robust on {p}')

    except Exception as e:
        LOG.error(e)


@dataclass
class _stage_25_Result:
    map_file_name: str
    paths_file_name: str
    adv_agent_id: int
    adv_agent_ds: int
    robust_radius: int

    is_kamikaze_successful: bool = None
    kamikaze_cost: int = -1
    collision_target_agent_id: int = -1
    collision_step: int = -1
    collision_cell: str = ''

    kamikaze_visited_nodes: int = -1
    kamikaze_expanded_nodes: int = -1
    kamikaze_run_time_seconds: float = 0.0

    comment: str = ''

def _stage_25_kamikaze(kamikaze_on_robust_paths_dir, rc, p):
    LOG.info(f'STARTED kamikaze on {p}')
    parts = p.stem.split('-')
    org_path_name = parts[1]
    adv_agent_id = int(parts[2].split('_')[1])
    adv_agent_ds = int(parts[3].split('_')[1])
    robust_radius = int(parts[4].split('_')[1])

    lp = paths_serializer.load(p)

    result_row = _stage_25_Result(
        rc.map_file_name,
        p.name,
        adv_agent_id,
        adv_agent_ds,
        robust_radius
    )

    for robust_mode in [RobustPathMode.OFFLINE]:
        kamikaze_route_path = kamikaze_on_robust_paths_dir / f'kamikaze-{robust_mode.name}-{org_path_name}-agent_{adv_agent_id}-ds_{adv_agent_ds}-rr_{robust_radius}{p.suffix}'
        kamikaze_paths = None
        try:
            with benchmark_utils.time_it(f'Running kamikaze with robust_mode={robust_mode}') as ti:
                kp = kamikaze_planner.KamikazePlanner(
                    lp.grid,
                    lp.agents,
                    astar.Searcher,
                    lambda agnt: dict(
                        h_func=mapf_heuristics.get_good_manhatten_like_heuristic(agnt)
                    ),
                    robust_mode=robust_mode,
                    robust_radius=robust_radius
                )
                goal_state, plan_metadata = kp.find()

            assert  isinstance(goal_state, kamikaze_planner.KamikazeState)
            kamikaze_paths = goal_state.paths
            result_row.is_kamikaze_successful = True
            result_row.kamikaze_cost = goal_state.g()
            result_row.kamikaze_expanded_nodes = plan_metadata.expanded_states
            result_row.kamikaze_visited_nodes = plan_metadata.visited_states
            result_row.kamikaze_run_time_seconds = ti.getElapsed()

            collision = goal_state.get_collision()
            result_row.collision_target_agent_id = collision.target_agent_id
            result_row.collision_step = collision.step
            result_row.collision_cell = str(collision.cell)
        except kamikaze_planner.NotFoundError:
            result_row.is_kamikaze_successful = False
        except Exception as e:
            LOG.error(e, exc_info=True)
            result_row.comment = str(e)

        paths_serializer.dump(kamikaze_route_path, lp.agents, [kamikaze_paths[a.id] for a in lp.agents] if kamikaze_paths else None,
                              lp.grid, metadata=vars(result_row))

    LOG.info(f'FINISHED kamikaze on {p}')

    return result_row

@dataclass
class _stage_3_Result:
    map_file_name: str
    paths_file_name: str
    adv_agent_id: int
    adv_damage_steps: int
    ms_original_path: int

    ms_mdr_path: int = -1
    mdr_expanded_nodes: int = -1
    mdr_visited_nodes: int = -1
    mdr_run_time_seconds: float = 0

    comment: str = ''

def _stage_3_normal_mdr(mdr_on_normal_paths_dir, adv_agent_id, adv_agent_ds, rc, p):
    LOG.info(f'STARTED MDR on {p} with agent [{adv_agent_id}] and DS={adv_agent_ds}')
    lp = paths_serializer.load(p)
    adv_agent = [a for a in lp.agents if a.id == adv_agent_id][0]

    adv_agent.is_adversarial = True
    adv_agent.damage_steps = adv_agent_ds
    for a in lp.agents:
        if a is not adv_agent:
            a.is_adversarial = False
            a.damage_steps = 0

    paths = {a.id: a.path for a in lp.agents}
    original_makespan = mdr_finder.get_makespan(paths, lp.agents)
    mdr_route_path = mdr_on_normal_paths_dir / f'mdr-{p.stem}-agent_{adv_agent.id}-ds_{adv_agent_ds}{p.suffix}'
    mdr_paths = None
    try:
        with benchmark_utils.time_it(f'Running MDR for adv agent:{adv_agent.id}, ds: {adv_agent_ds}') as ti:
            mdrf = mdr_finder.MaxDamageRouteFinder(lp.grid, lp.agents, astar.Searcher, lambda agnt: dict(
                h_func=mapf_heuristics.get_good_manhatten_like_heuristic(agnt)))
            goal_state, mdr_run_info = mdrf.find()
            mdr_paths = goal_state.paths

        mdr_run_time = ti.getElapsed()
        mdr_makespan = mdr_finder.get_makespan(mdr_paths, lp.agents)

        result = _stage_3_Result(
            rc.map_file_name,
            p.name,
            adv_agent.id,
            adv_agent_ds,
            original_makespan,
            mdr_makespan,
            mdr_run_info.expanded_states,
            mdr_run_info.visited_states,
            mdr_run_time,
        )
    except Exception as e:
        LOG.error(e)
        result = _stage_3_Result(
            rc.map_file_name,
            p.name,
            adv_agent.id,
            adv_agent_ds,
            original_makespan,
            comment=str(e)
        )

    paths_serializer.dump(mdr_route_path, lp.agents, [mdr_paths[a.id] for a in lp.agents] if mdr_paths else None, lp.grid,
                          metadata=vars(result))
    return result

@dataclass
class _stage_4_Result:
    map_file_name: str
    paths_file_name: str
    adv_agent_id: int
    adv_agent_ds: int
    ms_original_normal_path: int
    ms_original_robust_path: int
    robust_radius: int

    # ms_mdr_online_path: int = -1
    # mdr_online_expanded_nodes: int = -1
    # mdr_online_visited_nodes: int = -1
    # mdr_online_run_time_seconds: float = 0.0
    #
    # ms_mdr_offline_path: int = -1
    # mdr_offline_expanded_nodes: int = -1
    # mdr_offline_visited_nodes: int = -1
    # mdr_offline_run_time_seconds: float = 0.0

    comment: str = ''


def _stage_4_robust_mdr(mdr_on_robust_paths_dir, rc, p):
    # TODO: only calculate robust MDR paths for paths where MDR had a big impact on the original path

    # robust_path-{p.stem}-agent_{adv_agent.id}-ds_{adv_agent_ds}{p.suffix}
    LOG.info(f'STARTED robust MDR on {p}')
    parts = p.stem.split('-')
    org_path_name = parts[1]
    adv_agent_id = int(parts[2].split('_')[1])
    adv_agent_ds = int(parts[3].split('_')[1])
    robust_radius = int(parts[4].split('_')[1])

    org_path_file_path = p.parent.parent / STAGE_01_NORMAL_PATHS / (org_path_name + p.suffix)
    org_lp = paths_serializer.load(org_path_file_path)
    org_makespan = org_lp.get_makespan()

    lp = paths_serializer.load(p)

    mdr_info = {}
    paths = {a.id: a.path for a in lp.agents}
    original_robust_makespan = mdr_finder.get_makespan(paths, lp.agents)

    result_row = _stage_4_Result(
        rc.map_file_name,
        p.name,
        adv_agent_id,
        adv_agent_ds,
        org_makespan,
        original_robust_makespan,
        robust_radius
    )

    for robust_mode in [RobustPathMode.ONLINE_CONST]:
        mdr_route_path = mdr_on_robust_paths_dir / f'mdr-{robust_mode.name}-{org_path_name}-agent_{adv_agent_id}-ds_{adv_agent_ds}-rr_{robust_radius}{p.suffix}'
        mdr_paths = None
        try:
            with benchmark_utils.time_it(f'Running MDR with robust_mode={robust_mode}') as ti:
                mdrf = mdr_finder.MaxDamageRouteFinder(
                    lp.grid,
                    lp.agents,
                    astar.Searcher,
                    lambda agnt: dict(
                        h_func=mapf_heuristics.get_good_manhatten_like_heuristic(agnt)
                    ),
                    robust_mode=robust_mode,
                    robust_radius=robust_radius
                )
                goal_state, mdr_run_info = mdrf.find()
                mdr_paths = goal_state.paths

            # ms_mdr_online_path: int = -1
            # mdr_online_expanded_nodes: int = -1
            # mdr_online_visited_nodes: int = -1
            # mdr_online_run_time_seconds: float = 0.0
            mdr_results = {
                f'ms_mdr_{robust_mode.name.lower()}_path': mdr_finder.get_makespan(mdr_paths, lp.agents),
                f'mdr_{robust_mode.name.lower()}_expanded_nodes': mdr_run_info.visited_states,
                f'mdr_{robust_mode.name.lower()}_visited_nodes': mdr_run_info.expanded_states,
                f'mdr_{robust_mode.name.lower()}_run_time_seconds': ti.getElapsed()
            }
            for k, v in mdr_results.items():
                setattr(result_row, k, v)
        except Exception as e:
            LOG.error(e, exc_info=True)
            result_row.comment = str(e)

        paths_serializer.dump(mdr_route_path, lp.agents, [mdr_paths[a.id] for a in lp.agents] if mdr_paths else None, lp.grid, metadata=vars(result_row))

    LOG.info(f'FINISHED robust MDR on {p}')

    return result_row



@argh.arg('-p', '--permutations', type=int, default=None)
@argh.arg('-r', '--random-seed', type=int, default=None)
def e2e_parallel(run_config_file_name: str, out_dir: str = None, random_seed=None, permutations=None,
                 map_file_name: str = None, max_adv_agent_ds: int=None):# int = 2):
    logging.getLogger('vgmapf.problems.mapf.multi_agent_pathfinding').setLevel(logging.INFO)

    cores_count = multiprocessing.cpu_count()

    LOG.info(f'Detected {cores_count} cores!')

    print(run_config_file_name)
    #rc1 = config.load('config-simple.yml')
    rc1 = config.load(run_config_file_name)
    if permutations is not None:
        rc1.permutations = permutations
    if map_file_name:
        rc1.map_file_name = map_file_name
    random.seed(random_seed)


    max_adv_agent_ds = rc1.robust_route


    if out_dir is None:
        out_dir = pathlib.Path(__file__).parent / 'outputs' / time_utils.get_current_time_stamp()
    else:
        out_dir = pathlib.Path(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_dir1 = out_dir

    grid = grid2d.Grid2D.from_file(pathlib.Path(rc1.map_file_name))
    _update_start_and_goal_cells(rc1, grid)

    print(grid.to_str(start=rc1.start, end=rc1.end))

    start_time = timeit.default_timer()

    max_agents = len(rc1.agents)
    for swarm_ammount in range(2, max_agents+1): #(max_agents, max_agents+1):
        rc = rc1
        # Make folders for number of agents
        swarm_out_dir = out_dir1 / str(len(rc.agents))
        swarm_out_dir.mkdir(parents=True, exist_ok=True)
        out_dir = swarm_out_dir


        # Stage 1 - build normal paths
        LOG.info('START 01 - building normal paths')

        normal_paths_dir = out_dir / STAGE_01_NORMAL_PATHS

        normal_paths_dir.mkdir(parents=True, exist_ok=True)

        tasks = [(normal_paths_dir, grid, rc, permutation_idx) for permutation_idx in range(rc.permutations)]

        with multiprocessing.Pool(processes=cores_count) as pool:
            pool.starmap(_stage_1_normal_paths, tasks)

        LOG.info('FINISH 01 - building normal paths')

        LOG.info('STARTED 02 - run Robust Routes on normal paths')

        # Stage 2 - robust routes

        robust_paths_dir = out_dir / STAGE_02_ROBUST_PATHS
        robust_paths_dir.mkdir(parents=True, exist_ok=True)

        tasks = [(robust_paths_dir, grid, max_adv_agent_ds, p) for p in normal_paths_dir.iterdir()]

        with multiprocessing.Pool(processes=cores_count) as pool:
            pool.starmap(_stage_2_normal_robust, tasks)

        LOG.info('FINISHED 02 - run Robust Routes on normal paths')


        # #Stage 25 - run kamikaze on robust routes
        # LOG.info('STARTED 025 - run kamikaze on robust routes')
        #
        # kamikaze_on_robust_paths_dir = out_dir / STAGE_025_KAMIKAZE
        # kamikaze_on_robust_paths_dir.mkdir(parents=True, exist_ok=True)
        #
        # kamikaze_on_robust_results_summary = kamikaze_on_robust_paths_dir / '025-kamikaze_on_robust_paths.csv'
        #
        # # noinspection DuplicatedCode
        # tasks = [(kamikaze_on_robust_paths_dir, rc, p) for p in robust_paths_dir.iterdir()]
        # with multiprocessing.Pool(processes=cores_count) as pool:
        #     results = pool.starmap(_stage_25_kamikaze, tasks)
        #
        # if results:
        #     with kamikaze_on_robust_results_summary.open('w', newline='') as fresults:
        #         out_csv = csv.DictWriter(fresults, vars(results[0]).keys())
        #         out_csv.writeheader()
        #
        #         for row in results:
        #             try:
        #                 out_csv.writerow(vars(row))
        #             except Exception:
        #                 LOG.warning(f'Failed writing row: {row}', exc_info=True)
        #
        #         fresults.flush()
        #
        # LOG.info('FINISHED 025 - run kamikaze on robust routes')

        # Stage 3 - run MDR on normal paths

        LOG.info('STARTED 03 - run MDR on normal paths')

        mdr_on_normal_paths_dir = out_dir / STAGE_03_MDR_NORMAL_PATHS
        mdr_on_normal_paths_dir.mkdir(parents=True, exist_ok=True)

        mdr_on_normal_results_summary = mdr_on_normal_paths_dir / '03-mdr_on_normal_paths-results.csv'

        #tasks = [
        #    (mdr_on_normal_paths_dir, adv_agent.id, adv_agent_ds, rc, p) for p in normal_paths_dir.iterdir() for adv_agent
        #    in
        #    paths_serializer.load(p).agents for adv_agent_ds in range(1, max_adv_agent_ds + 1)
        #]

        tasks = [
            (mdr_on_normal_paths_dir, adv_agent.id, adv_agent_ds, rc, p) for p in normal_paths_dir.iterdir() for
            adv_agent
            in
            paths_serializer.load(p).agents for adv_agent_ds in range(1, max_adv_agent_ds + 1)
        ]

        LOG.debug(f'stage_3 tasks:\n\t' + '\n\t'.join(str(x) for x in tasks))
        with multiprocessing.Pool(processes=cores_count) as pool:
            results = pool.starmap(_stage_3_normal_mdr, tasks)

        if results:
            with mdr_on_normal_results_summary.open('w', newline='') as fresults:
                out_csv = csv.DictWriter(fresults, vars(results[0]).keys())
                out_csv.writeheader()

                for row in results:
                    try:
                        out_csv.writerow(vars(row))
                    except Exception:
                        LOG.warning( f'Failed writing row: {row}', exc_info=True)

        LOG.info('FINISHED 03 - run MDR on normal paths')

        LOG.info('STARTED 04 - run MDR on robust paths')

        # Stage 4 - MDR on robust paths

        mdr_on_robust_paths_dir = out_dir / STAGE_04_MDR_ROBUST_PATHS
        mdr_on_robust_paths_dir.mkdir(parents=True, exist_ok=True)

        mdr_on_robust_results_summary = mdr_on_robust_paths_dir / '04-mdr_on_robust_paths-results.csv'

        tasks = [(mdr_on_robust_paths_dir, rc, p) for p in robust_paths_dir.iterdir()]
        with multiprocessing.Pool(processes=cores_count) as pool:
            results = pool.starmap(_stage_4_robust_mdr, tasks)

        if results:
            with mdr_on_robust_results_summary.open('w', newline='') as fresults:
                out_csv = csv.DictWriter(fresults, vars(results[0]).keys())
                out_csv.writeheader()

                for row in results:
                    try:
                        out_csv.writerow(vars(row))
                    except Exception:
                        LOG.warning(f'Failed writing row: {row}', exc_info=True)

                fresults.flush()

        end_time = timeit.default_timer()
        LOG.info(
            f'FINISHED 04 - run MDR on robust paths, elapsed:{end_time - start_time:2f} = {datetime.timedelta(seconds=end_time - start_time)}')
        del rc1.agents[-1]

def test():
    from IPython import embed;
    embed()


def _main():
    logutils.init_logging(f'logs/vgmapf.log')
    out = six.StringIO()

    argh.dispatch_commands(
        [obj for name, obj in inspect.getmembers(sys.modules[__name__]) if
         inspect.isfunction(obj) and obj.__module__ == '__main__' and not name.startswith('_')],
        output_file=out
    )

    try:
        print(out.getvalue())
    except Exception:
        pprint.pprint(out.getvalue())


if '__main__' == __name__:
    _main()
