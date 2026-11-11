from .base import SideTaskDefinition


def get_side_task(type: str, **kwargs) -> SideTaskDefinition:
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    if type == "rolling_mod_sum":
        from . import rolling_mod_sum
        return rolling_mod_sum.make_task(**kwargs)
    elif type == "limited_arithmetic":
        from . import limited_arithmetic
        return limited_arithmetic.make_task(**kwargs)
    elif type == "ball_and_cups":
        from . import ball_and_cups
        return ball_and_cups.make_task(**kwargs)
    elif type == "gl32":
        from . import gl32
        return gl32.make_task(**kwargs)
    elif type == "function_composition":
        from . import function_composition
        return function_composition.make_task(**kwargs)
    elif type == "prontoqa":
        from . import prontoqa
        return prontoqa.make_task(**kwargs)
    raise ValueError(f"Unknown side task type: {type}")
