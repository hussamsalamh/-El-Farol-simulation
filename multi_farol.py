from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from typing import List


class Strategy:
    def __init__(self, id, M) -> None:
        self.id = id
        self.M = M
        self.weights = np.random.uniform(-1, 1, size=M)
        # last M predictions to be used in score calculation
        self.memory = np.zeros(M)

    def calculate_score(self, attendance: List[int]):
        """
            Calculate score of the strategy
        """
        return np.sum(np.abs(self.memory - attendance[-self.M:][::-1]))

    def calculate_prediction(self, attendance: List[int]):
        """
            Calculate prediction of the strategy
        """
        P = np.dot(self.weights, attendance[-self.M:][::-1])
        # update predictions
        self.memory = np.roll(self.memory, 1)
        self.memory[0] = P
        return P

class Constraint:
    """
        Constraint is defined as a tuple of threshold T, # of stratiges K, memory size M and penalt value p

        If an agent does not satisfy the constraint, it recieves penalty p

        Each agent has a set of strategies for each constraint

        In order for an agent to attend, it should have total penalty below certain threshold

    """
    def __init__(self, T, K, M, p) -> None:
        self.T = T
        self.K = K
        self.M = M
        self.p = p

        self.strategies = [Strategy(i, M) for i in range(K)]

    def satified(self, attendance: List[int]):
        """
            1. Calculate score for each strategy
            2. Choose strategy with lowest score
            3. Calculate prediction for chosen strategy
            4. Attend if prediction is less than threshold
        """
        scores = np.zeros(self.K)
        predictions = np.zeros(self.K)
        for i, strategy in enumerate(self.strategies):
            scores[i] = strategy.calculate_score(attendance)
            predictions[i] = strategy.calculate_prediction(attendance)

        i_min = np.argmin(scores)
        return predictions[i_min] < self.T

class Agent:
    def __init__(self, id, Ts: List[int], Ks: List[int], Ms: List[int], ps: List[int], P: int) -> None:
        """
            Ts: list of thresholds
            Ks: list of number of strategies
            Ms: list of memory size
            ps: list of penalty values

            P: penalty threshold
        """
        self.id = id
        self.Ts = Ts
        self.Ks = Ks
        self.Ms = Ms
        self.ps = ps
        self.P = P

        self.constraints = [Constraint(T, K, M, p) for T, K, M, p in zip(Ts, Ks, Ms, ps)]

    def should_attend(self, attendance: List[int]):
        """
            1. Calculate penalty for each constraint
            2. Attend if total penalty is less than threshold
        """
        penalty = 0
        for constraint in self.constraints:
            if not constraint.satified(attendance):
                penalty += constraint.p

        return penalty <= self.P


class ElFarol:
    """
        El Farol Bar Problem, with multiple constraints
    """

    def __init__(self, N: int, Ts: List[int], Ks: List[int], Ms: List[int], ps: List[int], P: float, animate=False) -> None:
        self.N = N
        self.Ts = Ts
        self.Ks = Ks
        self.Ms = Ms
        self.animate = animate
        ones = int(self.N * P)
        zeros = self.N - ones 
        self.agents = [Agent(i, Ts, Ks, Ms, ps, 1) for i in range(ones)]
        for _ in range(zeros):
            self.agents.append(Agent(_, Ts, Ks, Ms, ps, 0))
        self.attendance = [0] * max(Ms)  # initial attendance
        self.steps = 0

    def step(self):
        """
            1. Calculate attendance for each agent
            2. Update attendance
        """
        self.steps += 1
        attending = 0
        for agent in self.agents:
            if agent.should_attend(self.attendance):
                attending += 1

        print(f'\rStep {self.steps}: {attending} attending', end='')
        self.attendance.append(attending)


    def run(self, n_steps: int):
        """
            Run simulation for n_steps
        """
        if self.animate:
            fig, ax = plt.subplots()
            ax.set_xlim(0, n_steps)
            ax.set_ylim(0, self.N)
            line, = ax.plot([], [], lw=2)
            def init():
                line.set_data([], [])
                return line,

            def animate(i):
                self.step()
                x = np.arange(0, i+1)
                y = self.attendance
                line.set_data(x, y)
                return line,

            anim = FuncAnimation(fig, animate, init_func=init,
                                frames=n_steps, interval=20, blit=True)
            plt.show()
        else:
            for _ in range(n_steps):
                self.step()

    def plot(self):
        """
            Plot attendance
        """
        plt.figure()
        plt.plot(self.attendance[max(self.Ms):])
        # line at T
        for t in set(self.Ts):
            plt.axhline(y=t, color='r', linestyle='--')

        plt.axhline(y=np.average(self.attendance[max(self.Ms):]), color='g', linestyle='--',
                    linewidth=1, label='avg = {:.2f}'.format(np.average(self.attendance[max(self.Ms):])))
        plt.xlabel('Steps')
        plt.ylabel('Attendance')
        plt.title(f'N: {self.N}, T: {self.Ts}, K: {self.Ks}, M: {self.Ms}')
        plt.legend()

        plt.ylim(0, self.N)
        # plt.savefig(f'N{self.N}_T{self.T}_K{self.K}_M{self.M}.png')
        # plt.close()
        plt.show()


def main():
    N = 100  # number of agents
    Ts = [30, 60]  # thresholds
    Ks = [5, 5]  # number of strategies
    Ms = [50, 50]  # memory size
    ps = [1, 1]  # penalty values
    # P = .9

    AVG = []
    n_steps = 100  # number of steps
    for P in np.arange(0, 1, 0.05):
        el_farol = ElFarol(N, Ts, Ks, Ms, ps, P, animate=False)
        el_farol.run(n_steps)
        avg = np.average(el_farol.attendance[max(Ms):])
        print(f'P: {P}, avg: {avg}')
        AVG.append(avg)
        # el_farol.plot()
    plt.figure()
    plt.plot(np.arange(0, 1, 0.05), AVG)
    plt.ylim(0, N)
    plt.show()
if __name__ == "__main__":
    main()

    
