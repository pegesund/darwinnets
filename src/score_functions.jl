function softmax(neuralNet::NeuralNet)
    a = neuralNet.layers[length(neuralNet.layers)].values
    c = maximum(a)
    exp.(a .- c) / sum(exp.(a .- c))
end

function crossEntropyOneNumber(neuralNet::NeuralNet, number::Int)
    s = softmax(neuralNet)
    sSafe = s .+ 1e-8
    y = zeros(length(s))
    y[number + 1] = 1.0
    return -mean(y .* log.(sSafe))
end   

function mySoftMax(a::Array{Float64})
    c = maximum(a)
    exp.(a .- c) / sum(exp.(a .- c))    
end

function adaptedRmse(neuralNet::NeuralNet, number::Int)
    predictions = last(neuralNet.layers).values
    loss = 0.0
    for i in 1:length(predictions)
        loss += if i == number
            if predictions[i] < 0
                -predictions[i]
            else
                predictions[i]
            end    
        else
            abs(0 - predictions[i]) 
        end 
    end
    loss / length(predictions)
end
