@with_kw struct NeuralNetSettings
    growth_rate_default::Float64 = 0.01
    chance_growth_rate_increase::Integer = 3 # increase rate in heritage
    chance_growth_rate_direction::Integer = 4 # chanses a growth rate should change direction
    chance_singe_cell_mutation::Integer = 10 # chances for changes in activation function in layer, in 100000 th
    chance_activation_function::Integer = 10
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
    score::Float64 = 100
    generations::Int = 0
end


mutable struct NeuralNet
    layers::Array{Layer,1}
    bias::Array{Float64,1}
    params::NeuralNetSettings
    stats::Stats
end

