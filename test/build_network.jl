using Test

using DarwinNets: Layer, create_neuralnet, NeuralNet, print_all, relu, add_layer, new_layer, feed_forward, softmax, evolute
using PrettyPrint: pformat, pprint, pp_impl
using PrettyPrint
using BenchmarkTools



@test true


function PrettyPrint.pp_impl(io, m::Matrix{K}, indent)::Int where {K}
    print(io, m)
    return indent
end

network = create_neuralnet()

first_layer = new_layer([1, 200])
second_layer = new_layer(zeros(3); activation = relu)

add_layer(network, first_layer)
add_layer(network, second_layer)
add_layer(network, new_layer(zeros(3); activation = identity))

feed_forward(network)
println()

pprint(network)

network2 = evolute(network)
feed_forward(network2)

pprint(network2)
