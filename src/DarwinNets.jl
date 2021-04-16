module DarwinNets

# using Reviste

# train_x, train_y = MNIST.traindata()

# using ActivationFunctions

include("activation_functions.jl")
using Parameters


@with_kw struct NeuralNetSettings
    growth_rate_default::Float64 = 0.01
    chance_growth_rate_increase::Integer = 3 # increase rate in heritage
    chance_growth_rate_direction::Integer = 4 # chanses a growth rate should change direction
    chance_singe_cell_mutation::Integer = 10 # chances for changes in activation function in layer, in 100000 th
    chance_activation_function::Integer = 10
    chance_add_layer::Integer = 8
    chance_delete_layer::Integer = 8
    change_decrease_layer::Integer = 1
    change_increase_layer::Integer = 1
end


mutable struct Layer
    weights::Array{Float64,2}
    values::Array{Float64,1}
    activation
    direction::Array{Float64,2}
    growth_rate::Array{Float64,2}
end

mutable struct NeuralNet
    layers::Array{Layer,1}
    bias::Array{Float64,1}
    params::NeuralNetSettings
end

function create_neuralnet()
    neuralNet = NeuralNet(Layer[], Float64[],NeuralNetSettings())
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
        # neuralNet.layers[layerBelow].weights = rand(d2, d1)
        neuralNet.layers[layerBelow].weights = ones(d2, d1)

        # add direction and growth rate
        direction = zeros(d2, d1)
        growth_rate = zeros(d2, d1)
        for i in eachindex(direction)
            direction[i] = rand([-1,1])
            growth_rate[i] = neuralNet.params.growth_rate_default
        end
        neuralNet.layers[layerBelow].direction = direction
        neuralNet.layers[layerBelow].growth_rate = growth_rate
    else
        neuralNet.bias = ones(length(layer.values))
    end
end

function new_layer(values; activation = relu)
    Layer(Array{Float64}(undef, 0, 2), values, activation, Array{Float64}(undef, 0, 2), Array{Float64}(undef, 0, 2))
end

function feed_forward(neuralNet)
    for i in 1:length(neuralNet.layers)-1
        activation = neuralNet.layers[i].activation
        for j in 1:length(neuralNet.layers[i + 1].values)
            weights = neuralNet.layers[i].weights[:, j]
            values = neuralNet.layers[i].values
            if i == 1
                values = values .+ neuralNet.bias
            end
            neuralNet.layers[i+1].values[j] = activation(sum(weights .* values))
        end
    end
end


function softmax(neuralNet)
    a = neuralNet.layers[length(neuralNet.layers)].values
    c = maximum(a)
    exp.(a .- c) / sum(exp.(a .- c))
end


function mutate(neuralNet)

end

function evolute(neuralNetOriginal)
    neuralNet = deepcopy(neuralNetOriginal)
    for i in 2:length(neuralNet.layers)
        layer = neuralNet.layers[i]
        for j in 1:length(layer.weights)
            # check if we should change growth rate
            recalculate_weights = false
            if rand(1:neuralNet.params.chance_growth_rate_increase) == 1
                    layer.growth_rate[j] *= rand(0:1) + rand()
                    recalculate_weights = true
            end

            # maybe change direction
            if rand(1:neuralNet.params.chance_growth_rate_direction) == 1
                    layer.direction[j] = rand([-1,1])
                    recalculate_weights = true
            end

            # recalculate weights
            if recalculate_weights
                if layer.direction[j] == 1
                    layer.weights[j] *= (1 + layer.growth_rate[j])
                else
                    layer.weights[j] /= (1 + layer.growth_rate[j])
                end
            end
        end
    end
    return neuralNet
end

end # module
