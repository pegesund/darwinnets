function softmax(neuralNet)
    a = neuralNet.layers[length(neuralNet.layers)].values
    c = maximum(a)
    exp.(a .- c) / sum(exp.(a .- c))
end
