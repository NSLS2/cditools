from __future__ import annotations

from typing import Any


def ordered_dict_move_to_beginning(od: dict[str, Any], key: str) -> None:
    if key not in od:
        return

    value = od[key]
    items = [(k, v) for k, v in od.items() if k != key]
    od.clear()
    od[key] = value
    od.update(items)
