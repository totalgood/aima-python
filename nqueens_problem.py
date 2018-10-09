"""
N-Queens Problem as a Graph Search
"""
import numpy as np
import heapq
from itertools import chain


class PriorityQueue:
    """ A priority queue based on heapq: push returns the lowest value (highest priority)

    >>> values = list(zip(range(7,0,-1), sorted('abcdefg', reverse=True)))
    >>> values
    [(7, 'g'), (6, 'f'), (5, 'e'), (4, 'd'), (3, 'c'), (2, 'b'), (1, 'a')]
    >>> q = PriorityQueue(values)
    >>> q
    [(1, 'a'), (3, 'c'), (2, 'b'), (4, 'd'), (6, 'f'), (7, 'g'), (5, 'e')]
    >>> q.pop()
    (1, 'a')
    >>> q
    [(2, 'b'), (3, 'c'), (5, 'e'), (4, 'd'), (6, 'f'), (7, 'g')]
    >>> q.pop()
    (2, 'b')
    >>> print(repr(q))
    [(3, 'c'), (4, 'd'), (5, 'e'), (7, 'g'), (6, 'f')]
    >>> q.push((0, ''))
    >>> q
    [(0, ''), (4, 'd'), (3, 'c'), (7, 'g'), (6, 'f'), (5, 'e')]
    >>> q.peek()
    (0, '')
    >>> q[0]
    (0, '')
    >>> q[0]
    (0, '')
    """

    def __init__(self, values=()):
        self.heap = list(values)
        self.heapify(values)
        for s in dir(self.heap):
            if s.startswith('__') and not getattr(self, s, None):
                setattr(self, s, getattr(self.heap, s))

    def heapify(self, values=()):
        self.heap = list(values)
        return heapq.heapify(self.heap)

    def merge(self, *iterables):
        self.heap = heapq.merge(chain([self.heap], *iterables))

    def peek(self):
        return self[0]

    def pop(self):
        return heapq.heappop(self.heap)

    def push(self, value):
        return heapq.heappush(self.heap, value)

    def pushpop(self, value):
        return heapq.heappushpop(self.heap, value)

    def replace(self, value):
        return heapq.heapreplace(self.heap, value)

    def __repr__(self):
        return repr(self.heap)

    def __str__(self):
        return str(self.heap)


class NQueens:
    """ An NxN chessboard with N queens that are each "safe" from attack from all others """

    def __init__(self, n=9):
        self.frontier = list()
        self.n = n
        # 3 redundant representations
        self.state = np.array([np.nan] * n)
        self.sorted_state = np.array(sorted(self.state))
        self.grid = np.zeros((self.n, self.n))
        self.frontier = PriorityQueue()

    def isgoal(self, state):
        state = state or self.state
        return self.isvalid() and None not in self.state

    def isvalid(self, state=None):
        if state is None:
            self.update_state(state)
        if np.isnan(self.state.sum()):
            return False
        if any(self.sorted_state != np.array(list(range(self.n)))):
            return False
        for i in range(int(self.n / 2)):
            if np.diag(self.grid, i).sum() > 1:
                return False
            if np.diag(self.grid, -i).sum() > 1:
                return False
        return True

    def heuristic(self):
        ones = np.ones(self.n)
        s = (self.grid.sum(axis=0) - ones).abs().sum()
        s += (self.grid.sum(axis=1) - ones).abs().sum()
        for i in range(int(self.n / 2)):
            s += (np.diag(self.grid, i) - 1).abs().sum()
            s += (np.diag(self.grid, -i) - 1).abs().sum()
        return s

    def place_queen(self, i, j):
        self.state[i] = j
        self.update_state()

    def remove_queen(self, i, j):
        self.state[i] = np.nan
        self.update_state()

    def sorted_grid(self, state):
        sorted_state = np.array(sorted(self.state))
        grid = np.zeros((self.n, self.n))
        for i, j in enumerate(self.state):
            grid[i, j] = 1
        return sorted_state, grid

    def update_state(self, state=None):
        self.sorted_state, self.grid = self.sorted_grid(state or self.state)

    def expand_frontier(self):
        empty_cols = np.arange(self.n)[self.grid.sum(axis=0)]
        empty_rows = np.arange(self.n)[self.grid.sum(axis=1)]
        for i in empty_cols:
            for j in empty_rows:
                state = self.state
                state[i] = j
                self.frontier.push((self.heuristic(state), state))

    def __repr__(self):
        return repr(self.heap)

    def __str__(self):
        return str(self.heap)
