import textwrap
import json
import h5py
from .io.parameters import get_params

INDENT = "  "
INDENT_GROUP = "| "
MAX_NUMERIC_MEMBERS = 3


def all_numeric_keys(gr: h5py.Group):
    return all(k.isnumeric() for k in gr.keys())


def str_dataset(ds: h5py.Dataset):
    return repr(ds)


def str_group(gr: h5py.Group | h5py.File, keys: list[str] | None = None):
    out = []
    if keys is None:
        keys = sorted(gr.keys(), key=str)
    for name in keys:
        obj = gr[name]
        if isinstance(subgroup := obj, h5py.Group):
            if len(subgroup) > MAX_NUMERIC_MEMBERS and all_numeric_keys(subgroup):
                # Numeric indexed group, show only a few examples and the size
                subgroup_keys = sorted(subgroup.keys(), key=int)
                out.append(
                    f"{name} <{len(subgroup)} members: "
                    + (
                        f"[{subgroup_keys[0]} ... {subgroup_keys[-1]}]"
                        if len(subgroup) > 10
                        else f"[{', '.join(subgroup_keys)}]"
                    )
                    + ">:"
                )
                out.append(
                    textwrap.indent(
                        str_group(subgroup, keys=subgroup_keys[:MAX_NUMERIC_MEMBERS]),
                        INDENT_GROUP,
                    )
                )
            else:
                out.append(f"{name}:")
                out.append(textwrap.indent(str_group(subgroup), INDENT_GROUP))
            continue
        if name == "parameters":
            out.append(f"parameters <json>:")
            out.append(
                textwrap.indent(
                    json.dumps(get_params(gr), indent=2),
                    INDENT,
                )
            )
            continue
        if isinstance(ds := obj, h5py.Dataset):
            out.append(f"{name}: {str_dataset(ds)}")
            # out.append(textwrap.indent("", INDENT))
        else:
            out.append(f"{obj.name}:")
            out.append(textwrap.indent(f"Unknown element: {obj!r}", INDENT))
    return "\n".join(out)


def str_datafile(file: h5py.File | str):
    # Convenience recursive call to allow filename as well:
    if isinstance(file, str):
        with h5py.File(file, mode="r") as file:
            return str_datafile(file)
    return str_group(file)


if __name__ == "__main__":
    import sys
    import os
    import glob

    filenames = sorted(
        set([os.path.abspath(p) for arg in sys.argv[1:] for p in glob.glob(arg)])
    )

    for n, filename in enumerate(filenames):
        if n > 0:
            print("-" * 72)
        print(f"\n{n+1}) {filename}\n")
        print(str_datafile(filename))
