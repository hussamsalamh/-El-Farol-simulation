from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from typing import List


alpha = 0

class Strategy:
    def __init__(self, id, M) -> None:
        self.id = id
        self.M = M
        self.weights = np.random.uniform(-1, 1, size=M)
        self.memory = np.zeros(M)  # last M predictions to be used in score calculation

    def calculate_score(self, attendance: List[int]):
        """
            Calculate score of the strategy
        """
        return np.sum(np.abs(self.memory - attendance[-self.M:][::-1]))

    def calculate_prediction(self, attendance: List[int]):
        """
            Calculate prediction of the strategy
        """
        P =  np.dot(self.weights, attendance[-self.M:][::-1])
        # update predictions
        self.memory = np.roll(self.memory, 1)
        self.memory[0] = P
        return P


class Agent:
    def __init__(self, id, T, K, M) -> None:
        self.id = id
        self.T = T  # threshold
        self.K = K  # number of strategies
        self.M = M  # memory size

        self.strategies = [Strategy(i, M) for i in range(K)]

    def should_attend(self, attendance: List[int]):
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

        global alpha
        i_min = np.argmin(scores)
        # if predictions[i_min] < self.T:
        # Update weights for other strategies, learning rate alpha is related to how accurate the prediction was
        # alpha = 0.05  # learning rate
        # beta = 0.01 # learning variance
        j_min = np.argmin(np.abs(predictions - self.T))
        for i, strategy in enumerate(self.strategies):
            if i != j_min:
                strategy.weights = strategy.weights + alpha * (strategy.weights - self.strategies[j_min].weights)
                    
        return predictions[i_min] < self.T

class ElFarol:
    """
    El Farol Bar Problem
    
    Parameters
    ----------
    N : int
        Number of agents
    T: int
        Threshold - Bar capacity
    K: int
        Number of strategies
    M: int
        Memory size

    Statitics
    ---------
    steps: int
        Number of steps until convergence
    attendance: List[int]
        List of attendance in each step
    """

    def __init__(self, N: int, T: int, K: int, M: int, animate: bool):
        self.N = N
        self.T = T
        self.K = K
        self.M = M
        self.animate = animate

        self.agents = [Agent(i, T, K, M) for i in range(N)]

        self.attendance = [T] * M
        self.steps = 0 # steps until convergence

        if self.animate:
            self.setup_animation()

    
    def step(self):
        """
            1. Cacluate number of agents that will attend
            2. update attendance
        """
        self.steps += 1
        attend = 0
        for agent in self.agents:
            if agent.should_attend(self.attendance):
                attend += 1

        self.attendance.append(attend)

    def run(self):
        """
            Run simulation until convergence
        """
        while self.steps < 200:
            print(f'\rStep: {self.steps}', end='', flush=True)
            self.step()

            # if self.attendance[-1] == self.attendance[-2]:
            #     break

    def setup_animation(self):
        """
            Setup animation
        """
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(0, self.steps)
        self.ax.set_ylim(0, self.N)
        self.line, = self.ax.plot([], [], lw=2)
        self.anim = FuncAnimation(self.fig, self.update_anim, frames=200, interval=20, blit=True)

    def update_anim(self, *args):
        """
            Update animation
        """
        if self.steps > 200:
            return self.line,
        self.step()
        print(f'\rStep: {self.steps}', end='', flush=True)
        # self.line.set_data(np.arange(self.steps), self.attendance[self.M:])
        self.ax.clear()
        # self.ax.set_xlim(0, self.steps)
        self.ax.set_ylim(0, self.N)
        self.line, = self.ax.plot(self.attendance, lw=2)

        self.ax.axhline(y=self.T, color='r', linestyle='--')
        self.ax.set_xlabel('Steps')
        self.ax.set_ylabel('Attendance')
        return self.line,

    def run_animation(self):
        """
            Run animation
        """
        # self.setup_animation()
        # anim = FuncAnimation(self.fig, self.animate, frames=self.steps, interval=20, blit=True)
        # self.anim = FuncAnimation(self.fig, self.animate, frames=self.steps, interval=200, blit=True)
        plt.show()

    def plot(self):
        """
            Plot attendance
        """
        global alpha
        plt.figure()
        plt.plot(self.attendance[self.M:])
        # line at T
        plt.axhline(y=self.T, color='r', linestyle='-')
        plt.xlabel('Steps')
        plt.ylabel('Attendance')
        plt.title(f'N: {self.N}, T: {self.T}, K: {self.K}, M: {self.M}')

        plt.ylim(0, self.N)
        plt.savefig(f'N{self.N}_T{self.T}_K{self.K}_M{self.M}_alpha{alpha:.2f}.png')
        plt.close()
        # plt.show()

    def print_stats(self):
        """
            Print statistics
        """
        print()
        avg = np.mean(self.attendance[self.M:])
        std = np.std(self.attendance[self.M:])
        print(f'N: {self.N}, T: {self.T}, K: {self.K}, M: {self.M}')
        print(f'Steps: {self.steps}')
        print(f'Avg: {avg} +- {std}')

        return avg, std

    

def main():
    N = 100
    T = 60
    MM = np.arange(1, 16)
    M = 5
    K = 15
    # KK = np.arange(1, 16)
    ALPHA = np.arange(0, 0.9, 0.05)
    AVG = []
    STD = []
    global alpha
    for alpha in ALPHA:
    # for M in MM:
        # K = 10
        # M = 5

        el_farol = ElFarol(N, T, K, M, animate=False)

        # el_farol.run_animation()
        el_farol.run()
        el_farol.plot()
        avg, std = el_farol.print_stats()
        AVG.append(avg)
        STD.append(std)

    plt.figure()
    plt.errorbar(ALPHA, AVG, yerr=STD, fmt='o')
    plt.xlabel('M')
    plt.ylabel('Avg Attendance')
    plt.ylim(0, N)
    plt.title(f'N: {N}, T: {T}, M: {M}')
    plt.savefig(f'N{N}_T{T}_K{K}_M{M}_avg.png')
    plt.close()

    print(np.average(AVG))


if __name__ == "__main__":
    main()

