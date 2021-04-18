module DarwinNets

# using Reviste

using MLDatasets
using Parameters
using Statistics
using StatsBase
using DataStructures


include("activation_functions.jl")
include("structures.jl")
include("score_functions.jl")
include("mnist.jl")


function create_neuralnet()
    neuralNet = NeuralNet(Layer[], Float64[],NeuralNetSettings(), Stats())
end

function add_layer(neuralNet, layer)
    push!(neuralNet.layers, layer)
    if length(neuralNet.layers) > 1
        # now add weights to layer below
        layerBelow = length(neuralNet.layers) - 1
        d1 = length(layer.values)
        d2 = length(neuralNet.layers[layerBelow].values)
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


function runBeforeKeep(network::NeuralNet, dataset::DataSet)
    for i in 1:network.params.batch_run_before_keep
        datasetTrainLength = length(dataset.test_y)
        for i in 1:trunc(Int, datasetTrainLength / ecoSystem.batch_size)
            batchIds = sample(1:datasetTrainLength, ecoSystem.batch_size, replace = false)
            batch = map(id -> dataset.test_x[id], batchIds)
        end
    end    
end

function runEcosystem(eva::NeuralNet, dataset::DataSet, ecoSystem::EcoSystem)
    allNets =  MutableBinaryMaxHeap{NeuralNet}()
    newNets = foldl((a, o) -> push!(a, mutate(eva)), 1:9; init = [])
    for epoch in 1:ecoSystem.epochs
    end
    println(length(newNets))
end

end # module
