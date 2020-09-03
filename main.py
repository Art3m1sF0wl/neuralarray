from numpy import exp, array, random, dot
from random import randint


class NeuralNetwork():
    def __init__(self):
        random.seed()

        self.synaptic_weights = 2 * random.random((3, 1)) - 1

    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in range(number_of_training_iterations):
            output = self.think(training_set_inputs)

            error = training_set_outputs - output

            adjustment = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))

            self.synaptic_weights += adjustment

    def think(self, inputs):
        return self.__sigmoid(dot(inputs, self.synaptic_weights))


if __name__ == "__main__":

    neural_network = NeuralNetwork()

    print ("Random starting synaptic weights: ")
    print (neural_network.synaptic_weights)

    inputs1 = array([[random.randint(0, 2), random.randint(0, 2), random.randint(0, 2)], [random.randint(0, 2), random.randint(0, 2), random.randint(0,2)], [random.randint(0, 2), random.randint(0, 2), random.randint(0, 2)], [random.randint(0, 2), random.randint(0, 2), random.randint(0, 2)]])
    outputs1 = array([[random.randint(0, 2), random.randint(0, 2), random.randint(0, 2), random.randint(0, 2)]]).T
    training_set_inputs = inputs1
    training_set_outputs = outputs1
    print ("int_value ->\n",inputs1)
    print ("out_value ->\n",outputs1)
    
    neural_network.train(training_set_inputs, training_set_outputs, 100000)

    print ("New synaptic weights after training: ")
    print (neural_network.synaptic_weights)

    result=array([random.randint(0, 2), random.randint(0,2), random.randint(0, 2)])
    print ("Considering new situation [", result[0],",", result[1],",", result[2],"] -> ?: ")
    print (neural_network.think(result))
