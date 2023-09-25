import numpy as np
from time import perf_counter
from numpy.linalg import norm
from typing import Dict

def squared_norm(x, **kwargs):
    return np.sum(np.square(x), axis=-1, **kwargs)

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
    path_deltas = np.diff(path_discretized, axis=0)
    hs = []
    for skeleton in skeletons.values():
        bs = skeleton_field_dir(skeleton, path_discretized[:-1])
        # Integrate the field along the path
        h = np.sum(np.sum(bs * path_deltas, axis=-1), axis=0)
        # round to nearest integer since the output should really either be 0 or 1
        # absolute value because we don't care which "direction" the loop goes through an obstacle
        h = abs(int(h.round(0)))
        hs.append(h)
    return tuple(hs)


def skeleton_field_dir(skeleton, r):
    """
    Computes the field direction at the input points, where the conductor is the skeleton of an obstacle.
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
    if not np.all(skeleton[0] == skeleton[-1]):
        raise ValueError("Skeleton must be a closed loop! Add the first point to the end.")

    s_prev = skeleton[:-1][None]  # [1, n, 3]
    s_next = skeleton[1:][None]  # [1, n, 3]

    p_prev = s_prev - r[:, None]  # [b, n, 3]
    p_next = s_next - r[:, None]  # [b, n, 3]
    squared_segment_lens = squared_norm(s_next - s_prev, keepdims=True)
    d = np.cross((s_next - s_prev), np.cross(p_next, p_prev)) / squared_segment_lens  # [b, n, 3]

    # bs is a matrix [b, n,3] where each bs[i, j] corresponds to a line segment in the skeleton
    squared_d_lens = squared_norm(d, keepdims=True)
    p_next_lens = norm(p_next, axis=-1, keepdims=True) + 1e-6
    p_prev_lens = norm(p_prev, axis=-1, keepdims=True) + 1e-6

    # Epsilon is added to the denominator to avoid dividing by zero, which would happen for points _on_ the skeleton.
    ε = 1e-6
    d_scale = np.where(squared_d_lens > ε, 1 / (squared_d_lens + ε), 0)

    bs = d_scale * (np.cross(d, p_next) / p_next_lens - np.cross(d, p_prev) / p_prev_lens)

    b = bs.sum(axis=1) / (4 * np.pi)
    return b


def discretize_path(path: np.ndarray, n=1000):
    """ densely resamples a path to one containing n points """
    num_points = path.shape[0]
    t = np.linspace(0, 1, num_points)
    t_new = np.linspace(0, 1, n)
    path_discretized = np.zeros((n, 3))

    for i in range(3):
        path_discretized[:, i] = np.interp(t_new, t, path[:, i])

    return path_discretized




def test_h():
    loop = np.array([
         [0,0,0],
         [1,0,0],
         [1,0,1],
         [0,0,1],
         [0,0,0]
    ])
    obs_loop =np.array([
        [0.5, -0.5, 0.5],
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 1.5],
        [0.5, -0.5, 1.5],
        [0.5, -0.5, 0.5]
    ])
    skeleton = {
        'obs': obs_loop,
    }

    t0 = perf_counter()
    h_sig = get_h_signature(loop, skeleton)
    print(h_sig)
    t1 = perf_counter()

    print(f"Time: {t1-t0:.4f}s")
    

if __name__ == '__main__':
    test_h()