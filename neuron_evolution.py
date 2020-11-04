'''
    Entry point to evolving the neural network.
    Start here
'''

from optimizer import Optimizer
from logger import logger
from tqdm import tqdm

class NeuronEvolution:
    def __init__(self, generations, population, params):
        self._generations = generations
        self._population = population
        self._params = params
        self._networks = None
        self.best_solutions = None

    def evolve(self, x_train, y_train, x_test, y_test, x_val, y_val):
        optimizer = Optimizer(self._params)
        self._networks = list(optimizer.create_population(self._population))

        for i in range(self._population - 1):
            self._train_networks(x_train, y_train, x_test, y_test, x_val, y_val)
            self._networks = optimizer.evolve(self._networks)
        
        self._networks = sorted(self._networks, key=lambda x: x.accuracy, reverse=True)
        # logging for 5 best solutions
        self.best_solutions = self._networks[:5]
        for solution in self.best_solutions:
            logger.info('Accuracy: {}, Parameters: {}'.format(solution.accuracy, solution.network))

    def _train_networks(self, x_train, y_train, x_test, y_test, x_val, y_val):
        pbar = tqdm(total=len(self._networks))
        for network in self._networks:
            network.train(x_train, y_train, x_test, y_test, x_val, y_val)
            pbar.update(1)
        pbar.close()

    def _get_average_accuracy(self, networks):
        return sum([network.accuracy for network in networks]) / len(networks)