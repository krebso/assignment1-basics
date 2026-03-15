from math import cos, pi


def cosine_lr_schedule(t, a_max, a_min, T_w, T_c) -> float:
    if t < T_w:
        return t / T_w * a_max
    if t < T_c:
        return a_min + 1 / 2 * (1 + cos((t - T_w) / (T_c - T_w) * pi)) * (a_max - a_min)

    return a_min
