import numpy as np
from pyh_signature import get_h_signature

def test_h():
    N = 10
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    tau_loop = np.stack([np.cos(angles), np.sin(angles), np.zeros(N)], axis=0)
    obs1_loop = np.stack([np.cos(angles), np.zeros(N), np.sin(angles)], axis=0)
    obs2_loop = np.stack([np.cos(angles), np.zeros(N), np.sin(angles)], axis=0) + 1
    skeleton = {
        'obs1': obs1_loop,
        'obs2': obs2_loop,
    }

    h_sig = get_h_signature(tau_loop, skeleton)

    assert h_sig == Multiset([[1, 0]])
    

if __name__ == '__main__':
    test_h()