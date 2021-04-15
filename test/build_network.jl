using Test

using DarwinNets: Layer, create_neuralnet, NeuralNet, print_all, relu, add_layer, new_layer, feed_forward, softmax
using PrettyPrint: pformat, pprint



@test true

network = create_neuralnet()

first_layer = new_layer([1, 200])
second_layer = new_layer(zeros(3); activation = relu)

add_layer(network, first_layer)
add_layer(network, second_layer)
add_layer(network, new_layer(zeros(3); activation = identity))

feed_forward(network)

pprint(network)
println()
println(sum(softmax(network)))

println("hupp")
