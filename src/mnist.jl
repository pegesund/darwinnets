function readMnist()
    dataset_x = []
    train_x, train_y = MNIST.traindata()
    c = convert(Array{Float64}, train_x[:, :, :])
    for i in 1:60000
        push!(dataset_x, reshape(c[:, :, i], 784))
    end
    dataset_y = convert(Array{Float64}, train_y)
    return DataSet(dataset_x, dataset_y)
end