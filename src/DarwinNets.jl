module DarwinNets

# using Reviste

# using ActivationFunctions

include("activation_functions.jl")

const growth_rate_default = 0.01
const growth_rate_increase = 10

mutable struct Layer
    weights::Array{Float64,2}
    values::Array{Float64,1}
    activation
    direction::Array{Float64,2}
    growth_rate::Array{Float64,2}
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
    if length(neuralNet.layers) > 1
        # now add weights to layer below
        layerBelow = length(neuralNet.layers) - 1
        d1 = length(layer.values)
        d2 = length(neuralNet.layers[layerBelow].values)
        neuralNet.layers[layerBelow].weights = rand(d2, d1)

        # add direction and growth rate
        direction = zeros(d2, d1)
        growth_rate = zeros(d2, d1)
        for i in eachindex(direction)
            direction[i] = rand(-1:1)
            growth_rate[i] = growth_rate_default
        end

    end
end

function new_layer(values; activation = relu)
    Layer(Array{Float64}(undef, 0, 2), values, activation, Array{Float64}(undef, 0, 2), Array{Float64}(undef, 0, 2))
end

function feed_forward(neuralNet)
    for i in 1:length(neuralNet.layers)-1
        for j in 1:length(neuralNet.layers[i + 1].values)
            weights = neuralNet.layers[i].weights[:, j]
            values = neuralNet.layers[i].values
            neuralNet.layers[i+1].values[j] = neuralNet.layers[i].activation(sum(weights .* values))
        end
    end
end


end # module
