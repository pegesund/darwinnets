@with_kw struct NeuralNetSettings
    growth_rate_default::Float64 = 0.01
    chance_growth_rate_increase::Integer = 200 # increase rate in heritage
    chance_growth_rate_direction::Integer = 400 # chanses a growth rate should change direction
    chance_singe_cell_mutation::Integer = 200 # chances for changes in activation function in layer, in 100000 th
    chance_activation_function::Integer = 100
    chance_add_layer::Integer = 8
    chance_delete_layer::Integer = 8
    change_decrease_layer::Integer = 10
    change_increase_layer::Integer = 10
end

mutable struct Layer
    weights::Array{Float64,2}
    values::Array{Float64,1}
    activation::Function
    direction::Array{Float64,2}
    growth_rate::Array{Float64,2}
end

@with_kw mutable struct Stats
    score::Float64 = typemax(Float64)
    generations::Int = 0
end


mutable struct NeuralNet
    layers::Array{Layer,1}
    bias::Array{Float64,1}
    params::NeuralNetSettings
    stats::Stats
end

struct DataSet
    train_x::Vector{Vector{Float64}}
    train_y::Vector{Int}
    test_x::Vector{Vector{Float64}}
    test_y::Vector{Int}
end

@with_kw mutable struct EcoSystem
    keep_number_of_children::Int = 3
    epochs::Int = 100000
    batch_size::Int = 30 # use 0 for all
    batch_run_before_keep::Int = 200
    deep_mutations = 50
end

isless(a::NeuralNet, b::NeuralNet) = isless(a.stats.score, b.stats.score)