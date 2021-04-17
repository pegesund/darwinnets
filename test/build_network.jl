

using Test
using Distributed

@everywhere using DarwinNets: Layer, create_neuralnet, NeuralNet, print_all, relu, add_layer, new_layer, feed_forward, softmax, evolute, mutate
using PrettyPrint: pformat, pprint, pp_impl
using PrettyPrint
using BenchmarkTools
using Serialization: serialize, deserialize



@test true

function many_evolute(nwork)
    for i in 1:100000
        nwork = evolute(nwork)
        feed_forward(nwork)
    end
    return nwork
end


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

mutate(network)

# @btime pprint(many_evolute(network))
