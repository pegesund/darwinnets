function softmax(neuralNet::NeuralNet)
    a = neuralNet.layers[length(neuralNet.layers)].values
    c = maximum(a)
    exp.(a .- c) / sum(exp.(a .- c))
end

function crossEntropy(neuralNet::NeuralNet, number::Int)
    s = softmax(neuralNet)
    sSafe = s .+ 1e-8
    y = zeros(length(s))
    y[number] = 1.0
    return -mean(y .* log.(sSafe))
end   