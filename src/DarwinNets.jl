module DarwinNets

# using Reviste

# using ActivationFunctions

include("activation_functions.jl")

mutable struct Layer
    weights::Array{Float64,2}
    values::Array{Float64,1}
    activation
end

struct NeuralNet
    layers::Array{Layer,1}
    bias::Array{Float64,1}
end

function create_neuralnet()
    neuralNet = NeuralNet(Layer[], Float64[])
end

function print_all()
    println(all_activation_functions)
end

function add_layer(neuralNet, layer)
    push!(neuralNet.layers, layer)
    # now add weights to layer below
    if length(neuralNet.layers) > 1
        layerBelow = length(neuralNet.layers) - 1
        d1 = length(layer.values)
        d2 = length(neuralNet.layers[layerBelow].values)
        neuralNet.layers[layerBelow].weights = rand(d1, d2)
    end
end

function new_layer(values; activation = relu)
    Layer(Array{Float64}(undef, 0, 2), values, activation )
end

function feed_forward(neuralNet)
    for layer in 0:length(neuralNet)
        print(layer)
    end
end


end # module
