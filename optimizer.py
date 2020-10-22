'''
    Class that holds a genetic algorithm for evolving a network.

    Credit:
    A lot of those code was originally inspired by:
    http://lethain.com/genetic-algorithms-cool-name-damn-simple/
'''

from functools import reduce
from operator import add
import random
from network import Network

class Optimizer():
    '''
    Class that implements genetic algorithm for MLP optimization
    '''
    def __init__(self, nn_param_choices, retain=0.4,
                random_select=0.1, mutate_chance=0.2):
        '''
        Create an optimizer

        Args:
            nn_param_choices (dict): possible network parameters
            retain (float): percentage of population to retain after each generation
            random_select (float): probability of a rejected network remaining in the population
            mutate_chance (float): probability a network will be randomly mutated
        '''
        self.mutate_chance = mutate_chance
        self.random_select = random_select
        self.retain = retain
        self.nn_param_choices = nn_param_choices

    def create_population(self, count):
        '''
        Create a population of random networks.

        Args:
            count (int): number of networks to generate, aka the size of the population

        Return: (list): population of network objects
        '''
        pop = []
        for _ in range(0, count):
            # create a random network
            network = Network(self.nn_param_choices)
            network.create_random()

            # add the network to our population
            pop.append(network)
        
        return pop

    @staticmethod
    def fitness(network):
        '''
        Return the accuracy, which is our fitness function
        '''
        return network.accuracy

    def grade(self, pop):
        '''
        Find average fitness for a population

        Args:
            pop (list): the population of networks

        Returns:
            (float): the average accuracy of the population
        '''
        summed = reduce(add, (self.fitness(network) for network in pop))
        return summed / float(len(pop))

    def breed(self, mother, father):
        '''
        Make two children as parts of their parents.

        Args:
            mother (dict): network parameters
            father (dict): network parameters

        Returns:
            (list): two network objects
        '''

        children = []
        for _ in range(2):
            child = {}

            # loop through the parameters and pick params for the kid
            for param in self.nn_param_choices:
                child[param] = random.choice(
                    [mother.network[param],
                    father.network[param]]
                )
            
            # now create a network object
            network = Network(self.nn_param_choices)
            network.create_set(child)

            # randomly mutate some of the children
            if self.mutate_chance > random.random():
                network = self.mutate(network)
            
            children.append(network)

        return children

    def mutate(self, network):
        '''
        Randomly mutate one part of the network

        Args:
            network (dict): the network parameters to mutate

        Returns:
            (Network): A randomly mutated network object
        '''

        # choose a random key
        mutation = random.choice(list(self.nn_param_choices.keys()))

        # mutate one of the params
        network.network[mutation] = random.choice(self.nn_param_choices[mutation])

        return network

    def evolve(self, pop):
        '''
        Evolve a population of networks

        Args:
            pop (list): a list of network parameters
        
        Returns:
            (list): the evolved population of networks
        '''

        # get scores for each network
        graded = [(self.fitness(network), network) for network in pop]

        # sort on the scores
        graded = [x[1] for x in sorted(graded, key=lambda x: x[0], reverse=True)]

        # get the number we want to keep for the next generation
        retain_length = int(len(graded) * self.retain)

        # the parents are every network we want to keep
        parents = graded[:retain_length]

        # for those we aren't keeping, randomly keep some anyway
        for individual in graded[retain_length:]:
            if self.random_select > random.random():
                parents.append(individual)

        # now find out how many spots we have left to fill
        parents_length = len(parents)
        desired_length = len(pop) - parents_length
        children = []

        # add children, which are bred from two remaining networks
        while len(children) < desired_length:

            # get random mom and dad
            male = random.randint(0, parents_length - 1)
            female = random.randint(0, parents_length - 1)

            # assuming they aren't the same network
            if male != female:
                male = parents[male]
                female = parents[female]

                # breed them
                babies = self.breed(male, female)

                # add the children one at a time
                for baby in babies:
                    # don't grow larger than desired length
                    if len(children) < desired_length:
                        children.append(baby)
        
        parents.extend(children)

        return parents