module DarwinNets

# using Reviste

# train_x, train_y = MNIST.traindata()
# convert(Array{Float64}, train_x[:, :, 1])
# convert(Array{Float64}, train_x[:, :, :])
# l.values = reshape(c[:, :, 1], 784)

# using ActivationFunctions

using MLDatasets

include("activation_functions.jl")
using Parameters
include("structures.jl")



function create_neuralnet()
    neuralNet = NeuralNet(Layer[], Float64[],NeuralNetSettings(), Stats())
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

function feed_forward(neuralNet, values)
    first(neuralNet.layers).values = values
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


function evolute(neuralNetOriginal)
    neuralNet = deepcopy(neuralNetOriginal)
    
    for i in 1:length(neuralNet.layers) - 1
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

function mutate(neuralNetOriginal)

    network = create_neuralnet()
    network.params = deepcopy(neuralNetOriginal.params)

    #first layer
    layer = first(neuralNetOriginal.layers)
    activation = layer.activation
    if rand(1:neuralNetOriginal.params.chance_activation_function) == 1
        activation = rand(all_activation_functions)
    end
    newLayer = new_layer(zeros(length(layer.values)); activation = activation)
    add_layer(network, newLayer)


    # middle layers
    for i in 2:length(neuralNetOriginal.layers) - 1
        layer = neuralNetOriginal.layers[i]
        layerLength = length(layer.values)
        addNumberLayer = 1
        if rand(1:neuralNetOriginal.params.change_decrease_layer) == 1
            layerLength +=  rand(1:10)
        end
        if rand(1:neuralNetOriginal.params.change_increase_layer) == 1
            layerLength -= rand(1:10)
        end
        if rand(1:neuralNetOriginal.params.change_increase_layer) == 1
            layerLength -= rand(1:10)
        end

        activation = layer.activation
        if rand(1:neuralNetOriginal.params.chance_activation_function) == 1
            activation = rand(all_activation_functions)
        end

        if rand(1:neuralNetOriginal.params.chance_add_layer) == 1
            addNumberLayer += 1
        end

        if rand(1:neuralNetOriginal.params.chance_delete_layer) == 1
            addNumberLayer += 1
        end

        for j in 1:addNumberLayer
            if layerLength > 2
                newLayer = new_layer(zeros(layerLength); activation = activation)
                add_layer(network, newLayer)
            end
        end

    end

    #last layer
    layer = last(neuralNetOriginal.layers)
    activation = layer.activation
    if rand(1:neuralNetOriginal.params.chance_activation_function) == 1
        activation = rand(all_activation_functions)
    end
    newLayer = new_layer(zeros(length(layer.values)); activation = activation)
    add_layer(network, newLayer)

    return network

end

function readMnist()
    dataset_x = []
    train_x, train_y = MNIST.traindata()
    c = convert(Array{Float64}, train_x[:, :, :])
    for i in 1:60000
        push!(dataset_x, reshape(c[:, :, i], 784))
    end
    dataset_y = convert(Array{Float64}, train_y)
    return (dataset_x, dataset_y)
end


function runEcosystem()
    allNets = []

end

end # module
