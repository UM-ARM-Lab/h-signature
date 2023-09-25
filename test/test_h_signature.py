import numpy as np
from time import perf_counter
from h_signature.h_signature_np import get_h_signature

def test_h():
    loop = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [1, 0, 1],
        [0, 0, 1],
        [0, 0, 0]
    ])
    obs_loop = np.array([
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
