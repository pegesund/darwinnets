using Test

using DarwinNets: Layer, create_neuralnet, NeuralNet, print_all, relu, add_layer, new_layer, feed_forward
using PrettyPrint: pformat, pprint



@test true

network = create_neuralnet()

first_layer = new_layer([1, 200])
second_layer = new_layer(zeros(3); activation = sin)

add_layer(network, first_layer)
add_layer(network, second_layer)

println(network)

feed_forward(network)
println(network)
