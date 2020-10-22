'''
    Entry point to evolving the neural network.
    Start here
'''

import logging
from optimizer import Optimizer
from tqdm import tqdm

# setup logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.DEBUG,
    filename='log.txt'
)

def train_networks(networks, dataset):
    '''
    Train each network.

    Args:
        networks (list): current population of networks
        dataset (str): dataset to use for training/evaluating
    '''

    pbar = tqdm(total=len(networks))
    for network in networks:
        network.train(dataset)
        pbar.update(1)
    
    pbar.close()

def get_average_accuracy(networks):
    '''
    Get the average accuracy fir a group of networks

    Args:
        networks (list): list of networks
    
    Returns:
        (float):  the average accuracy of a population of networks
    '''

    total_accuracy = 0
    for network in networks:
        total_accuracy += network.accuracy

    return total_accuracy / len(networks)

def generate(generations, population, nn_param_choices, dataset):
    '''
    Generate a network with genetic algorithm

    Args:
        generations (int): number of times to evolve the population
        population (int): number of networks in each population
        nn_param_choices (dict): parameter choices for networks
        dataset (str): dataset to use for training/evaluating
    '''

    optimizer = Optimizer(nn_param_choices)
    networks = optimizer.create_population(population)

    # evolve the generation
    for i in range(generations):
        logging.info('*** Doing generation %d of %d ***' % (i + 1, generations))

        # train and get accuracy for networks
        train_networks(networks, dataset)

        # get the average accuracy for this generation
        average_accuracy = get_average_accuracy(networks)

        # print out the average accuracy each generation
        logging.info('Generation average: %.2f%%' % (average_accuracy * 100))
        logging.info('-' * 80)

        # evolve, except on the last iteration
        if i != generations - 1:
            # do the evolution
            networks = optimizer.evolve(networks)

    
    # sort our final population
    networks = sorted(networks, key=lambda x: x.accuracy, reverse=True)

    # print out the top 5 networks
    print_networks(networks[:5])

def print_networks(networks):
    '''
    Print a list of networks

    Args:
        networks (list): the population of networks
    '''

    logging.info('-' * 80)
    for network in networks:
        network.print_network()

def main():
    '''
    Evolve a network
    '''

    generations = 10     # number of times to evolve the population
    population = 20     # number of networks in each generation
    dataset = 'mnist'

    nn_param_choices = {
        'nb_neurons': [64, 128, 256, 512, 768, 1024],
        'nb_layers': [1, 2, 3, 4],
        'activation': ['relu', 'elu', 'tanh', 'sigmoid'],
        'optimizer': ['rmsprop', 'adam', 'sgd', 'adagrad', 'adadelta', 'adamax', 'nadam']
    }

    logging.info('*** Evolving %d generations with population %d ***' % (generations, population))

    generate(generations, population, nn_param_choices, dataset)

if __name__ == '__main__':
    main()