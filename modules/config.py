"""Dataset configuration helpers."""

import os
from copy import deepcopy
from typing import Any, Dict, Optional


class TaskMode:
    WEIGHTED_EDGE_PREDICTION = 0
    WEIGHTED_GRAPH_COMPLETION = 1


_DATASET_CONFIGS: Dict[str, Dict[str, Any]] = {
    "V1": {
        "sample_file": "samples_DBLP_20241208.pt",
        "sample_file_completion": "samples_DBLP_20250119_V1_completion.pt",
        "base_path": "/f/code/IDEA_data",
        "data_subdir": "DBLP_data/dataset_filtered_20241127",
        "seq_length": 7,
        "valid_seq_length": 2,
        "device": "cuda:0",
        "device2": "cuda:1",
        "embedding_file": "DBLP_data/author_embeddings.npy",
        "embedding_multiple_times": True,
    },
    "V2": {
        "sample_file": "samples_DBLP_20241215.pt",
        "sample_file_completion": "samples_DBLP_20250119_V2_completion.pt",
        "base_path": "/f/code/IDEA_data",
        "data_subdir": "DBLP_data/dataset_filtered_20241215",
        "seq_length": 14,
        "valid_seq_length": 3,
        "device": "cuda:0",
        "device2": "cuda:0",
        "embedding_file": "DBLP_data/author_embeddings.npy",
        "embedding_multiple_times": True,
    },
    "IoT": {
        "sample_file": "samples_IoT_20250108.pt",
        "sample_file_completion": "samples_IoT_20250119_completion.pt",
        "base_path": "/f/code/IDEA_data",
        "data_subdir": "IoT_data",
        "seq_length": 84,
        "valid_seq_length": 10,
        "device": "cuda:0",
        "device2": "cuda:0",
        "embedding_file": "IoT_feat.npy",
        "embedding_multiple_times": False,
    },
    "WIDE": {
        "sample_file": "samples_WIDE_20250108.pt",
        "sample_file_completion": "samples_WIDE_20250119_completion.pt",
        "base_path": "/f/code/IDEA_data",
        "data_subdir": "WIDE_data",
        "seq_length": 740,
        "valid_seq_length": 10,
        "device": "cpu",
        "device2": "cpu",
        "embedding_file": "WIDE_feat.npy",
        "embedding_multiple_times": False,
    },
    "RW": {
        "sample_file": "samples_RW_20250109.pt",
        "sample_file_completion": "samples_RW_20250119_completion.pt",
        "base_path": "/f/code/IDEA_data",
        "data_subdir": "RW_data",
        "seq_length": 140,
        "valid_seq_length": 10,
        "device": "cuda:0",
        "device2": "cuda:0",
        "embedding_file": None,
        "embedding_multiple_times": False,
        "num_nodes": 1000,
    },
    "wikipedia": {
        "sample_file": "samples_wikipedia_20250409.pt",
        "sample_file_completion": "samples_wikipedia_20250409_completion.pt",
        "base_path": "/f/code/IDEA_data",
        "data_subdir": "wikipedia",
        "seq_length": int(300 * 0.7),
        "valid_seq_length": int(300 * 0.15),
        "device": "cpu",
        "device2": "cpu",
        "embedding_file": "../IDEA/data/wikipedia_node_feat.npy",
        "embedding_multiple_times": False,
    },
    "reddit": {
        "sample_file": "samples_reddit_20250409.pt",
        "sample_file_completion": "samples_reddit_20250409_completion.pt",
        "base_path": "/f/code/IDEA_data",
        "data_subdir": "reddit",
        "seq_length": int(300 * 0.7),
        "valid_seq_length": int(300 * 0.15),
        "device": "cpu",
        "device2": "cpu",
        "embedding_file": "../IDEA/data/reddit_node_feat.npy",
        "embedding_multiple_times": False,
    },
    "lastfm": {
        "sample_file": "samples_lastfm_20250409.pt",
        "sample_file_completion": "samples_lastfm_20250409_completion.pt",
        "base_path": "/f/code/IDEA_data",
        "data_subdir": "lastfm",
        "seq_length": int(300 * 0.7),
        "valid_seq_length": int(300 * 0.15),
        "device": "cpu",
        "device2": "cpu",
        "embedding_file": "../IDEA/data/lastfm_node_feat.npy",
        "embedding_multiple_times": False,
    },
}


def _validate_config(config: Dict[str, Any], task_mode: int) -> None:
    required_keys = {
        "sample_file",
        "sample_file_completion",
        "data_path",
        "seq_length",
        "valid_seq_length",
        "device",
        "device2",
        "embedding_multiple_times",
    }
    missing = sorted(required_keys - set(config))
    if missing:
        raise ValueError(f"Missing required configuration keys: {', '.join(missing)}")

    if task_mode == TaskMode.WEIGHTED_EDGE_PREDICTION:
        task_specific_keys = {"sample_file"}
    elif task_mode == TaskMode.WEIGHTED_GRAPH_COMPLETION:
        task_specific_keys = {"sample_file_completion"}
    else:
        raise ValueError("task must be one of 0, 1")

    missing_task_keys = sorted(task_specific_keys - set(config))
    if missing_task_keys:
        raise ValueError(
            f"Missing required configuration keys for task {task_mode}: {', '.join(missing_task_keys)}"
        )

    if config.get("embedding_file") is None and "num_nodes" not in config:
        raise ValueError("num_nodes is required when embedding_file is None")


def _resolve_path(base_path: Optional[str], relative_path: str, default_base: str) -> str:
    if os.path.isabs(relative_path):
        return relative_path
    chosen_base = base_path or default_base or "."
    return os.path.abspath(os.path.join(chosen_base, relative_path))


def get_dataset_config(version: str, task_mode: int, base_path: Optional[str] = None) -> Dict[str, Any]:
    """Return validated dataset configuration for a given version and task."""

    if version not in _DATASET_CONFIGS:
        raise ValueError('VERSION must be one of "V1", "V2", "IoT", "RW", "WIDE", "wikipedia", "reddit", "lastfm"')

    config = deepcopy(_DATASET_CONFIGS[version])
    default_base = config.pop("base_path", None)
    data_subdir = config.pop("data_subdir", None)

    if data_subdir is None:
        raise ValueError("data_subdir must be provided in dataset configuration")

    config["data_path"] = _resolve_path(base_path, data_subdir, default_base or "")
    embedding_file = config.get("embedding_file")
    if embedding_file is not None:
        config["embedding_file"] = _resolve_path(base_path, embedding_file, default_base or "")

    _validate_config(config, task_mode)
    return config

