from train import get_dataset
from neuron_evolution import NeuronEvolution

def main():
    path = r'E:\\GA-evolve-main\\dataset\\per_api_drebin_benign'
    x_train, y_train, x_test, y_test, x_val, y_val, original_test_y = get_dataset(path)

    generations = 10
    population = 20

    params = {
        'nb_neurons': [128, 256, 512, 1024],
        'nb_layers': [3, 4, 5],
        'activation': ['relu', 'sigmoid', 'tanh', 'elu'],
        'optimizer': ['adam', 'rmsprop', 'sgd', 'nadam', 'adagrad', 'adadelta', 'adamax']
    }

    neuron_evolution = NeuronEvolution(generations=generations, population=population, params=params)
    neuron_evolution.evolve(x_train, y_train, x_test, y_test, x_val, y_val)

if __name__ == '__main__':
    main()