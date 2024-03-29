relu(x) = max(0,x)
drelu(x) = x <= 0 ? 0 : 1
celu(x) = max(0,x)+min(0,α*(exp(x/α)-1))
sigmoid(x) = 1/(1 + exp(-x))


all_activation_functions = [relu, drelu, celu, sin, atanh, tanh, identity, sigmoid]
