from typing import List

import numpy as np
from l5kit.data import ChunkedDataset
from l5kit.dataset import AgentDataset

TRAIN_DSAMPLE_FRAMES = [49, 99, 149, 199]


def downsample_agents(
    zarr_dataset: ChunkedDataset,
    agent_dataset: AgentDataset,
    selected_frames: List[int] = [101],
) -> list:
    """
    Extract agents within selected frames from each scene.
    It like test_dataset.
    """
    scenes = zarr_dataset.scenes
    frames = zarr_dataset.frames

    selected_agent_ind_interval = []
    for selected_frame in selected_frames:
        frames_over_scenes = scenes["frame_index_interval"][:, 0] + selected_frame
        selected_agent_ind_interval.append(
            frames["agent_index_interval"][frames_over_scenes]
        )
    selected_agent_ind_interval = np.concatenate(selected_agent_ind_interval, axis=0)

    agent_ind_to_dataset_ind = {
        agent_ind: i
        for i, agent_ind in enumerate(np.nonzero(agent_dataset.agents_mask)[0])
    }

    selected_agents = []
    print("downsampling agents...")
    for start, end in selected_agent_ind_interval:
        possible_agents = np.arange(start, end)
        valid_agents = possible_agents[agent_dataset.agents_mask[possible_agents]]
        if valid_agents.shape[0] > 0:
            valid_agents_in_dataset_ind = [
                agent_ind_to_dataset_ind[ind] for ind in valid_agents
            ]
            selected_agents.append(valid_agents_in_dataset_ind)

    agents_list = np.concatenate(selected_agents).tolist()
    return agents_list


def print_argparse_arguments(p, bar: int = 50) -> None:

    """
    from https://qiita.com/ka10ryu1/items/0b24e39799b2457cba62
    Visualize argparse arguments.
    Arguments:
        p : parse_arge() object.
        bar : the number of bar on the output.

    """
    print("PARAMETER SETTING")
    print("-" * bar)
    args = [(i, getattr(p, i)) for i in dir(p) if "_" not in i[0]]
    for i, j in args:
        if isinstance(j, list):
            print("{0}[{1}]:".format(i, len(j)))
            [print("\t{}".format(k)) for k in j]
        else:
            print("{0:25}:{1}".format(i, j))
    print("-" * bar)
    return None
