def strict_bool(v):
    if v is None:
        return None

    if isinstance(v, bool):
        return v

    if isinstance(v, (int, float)):
        if v == 1:
            return True
        elif v == 0:
            return False
    elif isinstance(v, str):
        if v.lower() in ("yes", "true", "t", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "0"):
            return False

    return None