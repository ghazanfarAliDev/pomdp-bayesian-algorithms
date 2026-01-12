import numpy as np

class MCTSNode:
    """
    Stores statistics on edges (s,a):
      N[a] = visit count
      W[a] = total return
      Q(s,a) = W[a] / N[a]
    """
    def __init__(self, state, parent=None, action_taken=None):
        self.state = np.array(state, dtype=np.float32)
        self.parent = parent
        self.action_taken = action_taken

        self.children = {}   # action -> MCTSNode
        self.N = {}          # action -> visit count
        self.W = {}          # action -> total return
        self.is_terminal = False

    def q_value(self, a):
        if self.N.get(a, 0) == 0:
            return 0.0
        return self.W[a] / self.N[a]

    def is_fully_expanded(self, action_space_n):
        return len(self.children) == action_space_n
