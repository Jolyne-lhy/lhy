from rtchange.coding import SDNML
import math
import numpy as np

def zero2small(v, small=1e-5):
    return v if v > 0 else small

class ModifiedSDNML(SDNML):
    def length(self, x):
        """
        Rewrite function.
        """
        self._xs.pop(0)
        self._xs.append(x)
        xs = np.atleast_2d(self._xs).T
        self._update_stats(xs)
        score = 0
        if self._time > 1:
            score += (self._time)/2 * (
                math.log(zero2small((self._time)*self._stats['tau'])) -
                math.log(zero2small((self._time-1)*self._stats['prev_tau']))
            )
        self._time += 1
        return score

    def score(self, X):
        """
        Calculate anomaly score.
        X : array-like
        scores : anomaly score / change-point score
        """
        for x in X:
            yield self.length(x)