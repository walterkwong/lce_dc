
def create_filename(*args, **kwargs) -> str:
    parts = list(args)
    for name, value in kwargs.items():
        if isinstance(value, bool):
            if value:
                parts.append(name)
        elif isinstance(value, float):
            parts.append(f"{name}{value:.2f}")
        else:
            parts.append(f"{name}{value}")
    return "_".join(p for p in parts if p)
