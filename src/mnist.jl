function readMnist()
    dataset_train_x = []
    dataset_test_x = []
    train_x, train_y = MNIST.traindata()
    c = convert(Array{Float64}, train_x[:, :, :])
    for i in 1:60000
        push!(dataset_train_x, reshape(c[:, :, i], 784))
    end
    dataset_train_y = convert(Array{Int}, train_y)
    test_x, test_y = MNIST.testdata()
    c = convert(Array{Float64}, test_x[:, :, :])
    for i in 1:length(test_y)
        push!(dataset_test_x, reshape(c[:, :, i], 784))
    end
    dataset_test_y = convert(Array{Int}, test_y)

    return DataSet(dataset_train_x, dataset_train_y, dataset_test_x, dataset_test_y)
end