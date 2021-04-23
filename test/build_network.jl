

using Test
using Distributed

@everywhere using DarwinNets: Layer, create_neuralnet, NeuralNet, EcoSystem, relu, add_layer, new_layer, feed_forward, softmax, evolute, mutate, readMnist, crossEntropyOneNumber, runEcosystem, mySoftMax
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

#=
function PrettyPrint.pp_impl(io, m::Matrix{K}, indent)::Int where {K}
    print(io, m)
    return indent
end
=#



# @btime pprint(many_evolute(network))

network = create_neuralnet()


add_layer(network, new_layer(zeros(28 * 28)))
add_layer(network, new_layer(zeros(128); activation=relu))
add_layer(network, new_layer(zeros(10)))

# @btime many_evolute(networkLarge)
feed_forward(network, rand(28 * 28))

m =readMnist()

runEcosystem(network, m, EcoSystem())

#=
scores = [4.587812980390243e8, 1.4747939324384959e9, 
3.3220447877342606e8, 5.524774329184169e8, 
4.4096602050127804e8, 4.5410656869948006e8, 
4.029421116180034e8, 8.621092489694402e8, 
3.355029989437256e8, 3.850631947441855e8]



scores = relu.([4.587812980390243e8, 1.4747939324384959e9, 
3.3220447877342606e8, 5.524774329184169e8, 
4.4096602050127804e8, 4.5410656869948006e8, 
4.029421116180034e8, 8.621092489694402e8])


# scores = [-1.0, 0.0, 3.0, 5.0]

print(mySoftMax(scores))
=#