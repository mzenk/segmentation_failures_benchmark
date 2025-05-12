from typing import Any

from omegaconf import DictConfig, ListConfig


def flatten(cfg: Any, resolve: bool = False) -> list[tuple[str, Any]]:
    ret = []

    def handle_dict(key: Any, value: Any, resolve: bool) -> list[tuple[str, Any]]:
        return [(f"{key}.{k1}", v1) for k1, v1 in flatten(value, resolve=resolve)]

    def handle_list(key: Any, value: Any, resolve: bool) -> list[tuple[str, Any]]:
        return [(f"{key}.{idx}", v1) for idx, v1 in flatten(value, resolve=resolve)]

    if isinstance(cfg, DictConfig):
        for k, v in cfg.items_ex(resolve=resolve):
            if isinstance(v, DictConfig):
                ret.extend(handle_dict(k, v, resolve=resolve))
            elif isinstance(v, ListConfig):
                ret.extend(handle_list(k, v, resolve=resolve))
            else:
                ret.append((str(k), v))
    elif isinstance(cfg, ListConfig):
        for idx, v in enumerate(cfg._iter_ex(resolve=resolve)):
            if isinstance(v, DictConfig):
                ret.extend(handle_dict(idx, v, resolve=resolve))
            elif isinstance(v, ListConfig):
                ret.extend(handle_list(idx, v, resolve=resolve))
            else:
                ret.append((str(idx), v))
    else:
        raise TypeError("Expected config to be of type ListConfig or DictConfig (omegaconf)")

    return ret


def compare_configs(cfg1, cfg2, resolve=False):
    flat1 = dict(flatten(cfg1, resolve=resolve))
    flat2 = dict(flatten(cfg2, resolve=resolve))
    keys_only_in1 = set(flat1.keys()).difference(set(flat2.keys()))
    keys_only_in2 = set(flat2.keys()).difference(set(flat1.keys()))
    if len(keys_only_in1) > 0:
        print(f"Keys only in 1: {keys_only_in1}")
    if len(keys_only_in2) > 0:
        print(f"Keys only in 2: {keys_only_in2}")
    for k in flat1.keys():
        if flat1[k] != flat2[k]:
            print(f"DIFF {k}: {flat1[k]} != {flat2[k]}")
