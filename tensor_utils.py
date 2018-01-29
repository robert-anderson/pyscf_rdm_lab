import numpy as np
import itertools

def allclose_transpose(t1, t2, t_quit_on_first=True, atol=1e-8):
    assert(t1.shape==t2.shape)
    t = False
    for transpose in itertools.permutations(range(len(t1.shape))):
        if np.allclose(t1, t2.transpose(transpose), atol=atol):
            print transpose
            if t_quit_on_first:
                return True
            else:
                t = True
    return t
