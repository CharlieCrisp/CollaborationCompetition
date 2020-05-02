import abc
from collections import deque


class Solver(abc.ABC):
    @abc.abstractmethod
    def record_score(self, score):
        pass

    @abc.abstractmethod
    def is_solved(self) -> bool:
        pass


class AverageScoreSolver(Solver):
    def __init__(self, num_agents, solved_score, solved_score_period):
        self.latest_scores = deque(maxlen=solved_score_period)
        self.num_agents = num_agents
        self.solved_score = solved_score

    def record_score(self, score):
        self.latest_scores.append(score)

    def is_solved(self):
        return (sum(self.latest_scores) / len(self.latest_scores)) > self.solved_score
