"""Reading and writing pOwl parameters.

pOwl parameters are JSON-encoded dictionaries in datasets called 'parameters'.
"""
from __future__ import annotations as _annotations

import json
import h5py
from typing import TypeAlias, Union

GroupParams: TypeAlias = dict[
    str, Union[str, int, float, list[int | float], "GroupParams"]
]


def serialize(parameters: GroupParams) -> str:
    return json.dumps(parameters, separators=(",", ":"))


def deserialize(jsonstr: str) -> GroupParams:
    return json.loads(jsonstr)


def get_params(group: h5py.Group) -> GroupParams:
    if "parameters" in group:
        if isinstance(group["parameters"], h5py.Dataset):
            return deserialize(group["parameters"].asstr()[()])  # type: ignore
        raise ValueError(f"{group} has key 'parameters' but it's not a dataset.")
    return {}  # type: ignore


def save_params(parameters: GroupParams, group: h5py.Group) -> None:
    group.create_dataset(name="parameters", data=serialize(parameters))


def update_params(parameters: GroupParams, group: h5py.Group) -> GroupParams:
    old_params = get_params(group)
    parameters = {**old_params, **parameters}
    if parameters != old_params:
        del group["parameters"]
        save_params(parameters, group)
    return parameters
