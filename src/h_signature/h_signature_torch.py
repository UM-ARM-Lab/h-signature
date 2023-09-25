import torch
from torch.linalg import norm
from typing import Dict

def squared_norm(x, **kwargs):
    return torch.sum(torch.square(x), axis=-1, **kwargs)

def get_h_signature(path, skeletons: Dict):
    """
    Computes the h-signature of a path, given the skeletons of the obstacles.

    Args:
        path: A path through the environment, as a list of points in 3D.
        skeletons:  A dictionary of skeletons, where the keys are the names of the obstacles and the values are the
            skeletons of the obstacles.

    Returns:

    """
    # Densely discretize the path so that we can integrate the field along it
    path_discretized = discretize_path(path)
    path_deltas = torch.diff(path_discretized, axis=0)
    hs = []
    for skeleton in skeletons.values():
        bs = skeleton_field_dir(skeleton, path_discretized[:-1])
        # Integrate the field along the path
        h = torch.sum(torch.sum(bs * path_deltas, axis=-1), axis=0)
        # round to nearest integer since the output should really either be 0 or 1
        # absolute value because we don't care which "direction" the loop goes through an obstacle
        h = torch.abs(torch.round(h)).int()
        hs.append(h)
    return tuple(hs)


def skeleton_field_dir(skeleton, r):
    """
    Computes the field direction at the itorch.t points, where the conductor is the skeleton of an obstacle.
    A skeleton is defined by a set of points in 3D, like a line-strip, and can represent only a genus-1 obstacle (donut)
    Assumes μ and I are 1.
    Based on this paper: https://www.roboticsproceedings.org/rss07/p02.pdf

    Variables in my code <--> math in the paper:

        s_prev = s_i^j
        s_next = s_i^j'
        p_prev = p
        p_next = p'

    Args:
        skeleton: [n, 3] the points that define the skeleton
        r: [b, 3] the points at which to compute the field.
    """
    if not torch.all(skeleton[0] == skeleton[-1]):
        raise ValueError("Skeleton must be a closed loop! Add the first point to the end.")

    s_prev = skeleton[:-1][None]  # [1, n, 3]
    s_next = skeleton[1:][None]  # [1, n, 3]

    p_prev = s_prev - r[:, None]  # [b, n, 3]
    p_next = s_next - r[:, None]  # [b, n, 3]
    squared_segment_lens = squared_norm(s_next - s_prev, keepdims=True)
    d = torch.cross((s_next - s_prev), torch.cross(p_next, p_prev)) / squared_segment_lens  # [b, n, 3]

    # bs is a matrix [b, n,3] where each bs[i, j] corresponds to a line segment in the skeleton
    squared_d_lens = squared_norm(d, keepdims=True)
    p_next_lens = norm(p_next, axis=-1, keepdims=True) + 1e-6
    p_prev_lens = norm(p_prev, axis=-1, keepdims=True) + 1e-6

    # Epsilon is added to the denominator to avoid dividing by zero, which would happen for points _on_ the skeleton.
    ε = 1e-6
    d_scale = torch.where(squared_d_lens > ε, 1 / (squared_d_lens + ε), 0)

    bs = d_scale * (torch.cross(d, p_next) / p_next_lens - torch.cross(d, p_prev) / p_prev_lens)

    b = bs.sum(axis=1) / (4 * torch.pi)
    return b


def discretize_path(path: torch.tensor, n=1000):
    """ densely resamples a path to one containing n points """
    num_points = path.shape[0]
    t = torch.linspace(0, 1, num_points)
    t_tiled = t[:, None].repeat(1, n)
    t_new = torch.linspace(0, 1, n)
    next_i = (t_new > t_tiled).sum(axis=0)
    prev_i = next_i - 1

    p_prev = path[prev_i]
    p_next = path[next_i]
    weights = (t_new - t[prev_i]) / (t[next_i] - t[prev_i])
    weights[0] = 0
    path_discretized = p_prev + weights[:, None] * (p_next - p_prev)
    path_discretized[0] = path[0]
    path_discretized[-1] = path[-1] 

    return path_discretized