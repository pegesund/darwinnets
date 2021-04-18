

using Test
using Distributed

@everywhere using DarwinNets: Layer, create_neuralnet, NeuralNet, relu, add_layer, new_layer, feed_forward, softmax, evolute, mutate, readMnist, crossEntropyOneNumber
using PrettyPrint: pformat, pprint, pp_impl
using PrettyPrint
using BenchmarkTools
using Serialization: serialize, deserialize



@test true

function many_evolute(nwork)
    for i in 1:1000
        nwork = evolute(nwork)
    end
    return nwork
end


function PrettyPrint.pp_impl(io, m::Matrix{K}, indent)::Int where {K}
    print(io, m)
    return indent
end


# @btime pprint(many_evolute(network))

network = create_neuralnet()


add_layer(network, new_layer(zeros(28 * 28)))
add_layer(network, new_layer(zeros(128)))
add_layer(network, new_layer(zeros(128)))
add_layer(network, new_layer(zeros(10)))

# @btime many_evolute(networkLarge)
feed_forward(network, rand(28 * 28))

m =readMnist()

println(softmax(network))
println(crossEntropyOneNumber(network, 2))


