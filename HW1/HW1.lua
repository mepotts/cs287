-- Only requirement allowed
require("hdf5")

cmd = torch.CmdLine()

-- Cmd Args
cmd:option('-datafile', 'SST1.hdf5', 'data file')
cmd:option('-classifier', 'nb', 'classifier to use')

-- Hyperparameters
-- ...
cmd:option('-alpha', 0, 'smoothing value')
cmd:option('-lambda', 0, 'L2 regularization value')
cmd:option('-m', 100, 'size of minibatch for gradient descent')
cmd:option('-eta', 0.1, 'learning rate')
cmd:option('-epochs', 100, 'number of epochs for gradient descent')


-- predicts the class from the linear model specified by the parameters W and b
-- note: x is in sparse form
function predict_class(x, W, b)
    local y_hat = b:clone()
    local x = x[x:gt(1)]

    y_hat:add(W:index(2, x:long()):sum(2))

    local _, class = y_hat:max(1) -- get the argmax

    return class[1]
end

function eval_linear_model(W, b)
    local ncorrect = 0

    for i = 1, valid_input:size(1) do
        local x = valid_input[i]
        local class = predict_class(x, W, b)

        if class == valid_output[i] then
            ncorrect = ncorrect + 1
        end
    end

    print("# correct: ", ncorrect)
    print("# total: ", valid_input:size(1))
    print("Accuracy: ", ncorrect / valid_input:size(1), "\n")
end


function predict_test_classes(W, b)
    local predictions = torch.Tensor(test_input:size(1))

    for i = 1, test_input:size(1) do
        local x = test_input[i]
        local class = predict_class(x, W, b)

        predictions[i] = class
    end

    return predictions
end

function print_test_predictions(f, W, b)
    local f = f or io.stdout
    f:write("ID", ",", "Category", "\n")

    local predictions = predict_test_classes(W, b)
    for i = 1, predictions:size(1) do
        f:write(i, ",", predictions[i], '\n')
    end
end

function learn_naive_bayes(alpha)
    local alpha = alpha or 0

    -- Compute the prior
    local prior = torch.zeros(nclasses)
    for i = 1, train_output:size(1) do
        prior[train_output[i]] = prior[train_output[i]] + 1
    end
    prior:div(train_output:size(1))
    local b = torch.log(prior)

    -- Construct count matrix
    local F = torch.ones(nclasses, nfeatures):mul(alpha)
    for i = 1, train_input:size(1) do
        local class = train_output[i]
        local x = train_input[i][train_input[i]:gt(1)]

        if x:dim() ~= 0 then
            for j = 1, x:size(1) do
                local feature = x[j]
                F[class][feature] = F[class][feature] + 1
            end
        end
    end

    -- Compute the posterior by normalizing count matrix
    for i = 1, F:size(1) do
        local row = F:select(1, i)
        local row_sum = torch.sum(row)
        row:div(row_sum)
    end
    local W = torch.log(F)
    -- print(F[{{},{1,11}}])

    return W, b
end

function log_sum_exp(z)
    local M = z:max(1)[1]
    local ret = z:clone()
    ret:csub(M)
    return (torch.log(torch.exp(ret):sum(1)) + M)[1]
end

function softmax(x, W, b)
    local x = x[x:gt(1)]
    local z = b:clone()
    if x:dim() ~= 0 then
        z:add(W:index(2, x:long()):sum(2))
    end

    local denom = log_sum_exp(z)
    z:csub(denom)
    return torch.exp(z)
end

function multiclass_logistic_gradient(x, class, W, b)
    local y_hat = softmax(x, W, b)

    local grad = y_hat
    grad[class] = grad[class] - 1

    local grad_W = torch.zeros(nclasses, nfeatures)
    if x:dim() ~= 0 then
        for j = 1, x:size(1) do
            col = grad_W:select(2, x[j])
            col:add(grad)
        end
    end

    return grad_W, grad
end

function multiclass_logistic_loss(x, class, W, b)
    local loss = 0
    local y_hat = softmax(x, W, b)
    loss = loss + torch.log(y_hat[class])
    loss = -loss
    return loss
end

function minibatch_sgd(gradient, loss, lambda, m, eta, epochs)
    local W = torch.zeros(nclasses, nfeatures)
    local b = torch.zeros(nclasses)

    local n = 0
    while n < epochs do
        print("Iteration", n)
        local sample = torch.multinomial(torch.ones(train_input:size(1)), m, false)

        local grad_W_hat = torch.zeros(nclasses, nfeatures)
        local grad_b_hat = torch.zeros(nclasses)

        if loss then
            local L = 0
            for i = 1, 10000 do
                local x = train_input[i][train_input[i]:gt(1)]
                local class = train_output[i]
                L = L + loss(x, class, W, b)
            end
            L = L + lambda / 2 * (torch.sum(torch.pow(torch.norm(W, 2, 1), 2)) + torch.pow(torch.norm(b), 2))
            print("Est. loss", L)
        end

        for i = 1, sample:size(1) do
            local x = train_input[sample[i]][train_input[sample[i]]:gt(1)]
            local class = train_output[sample[i]]

            local grad_W, grad_b = gradient(x, class, W, b)

            grad_W_hat:add(grad_W:div(m))
            grad_b_hat:add(grad_b:div(m))
        end

        W:mul(1 - eta * lambda / train_input:size(1))
        b:mul(1 - eta * lambda / train_input:size(1))

        W:csub(grad_W_hat:mul(eta))
        b:csub(grad_b_hat:mul(eta))

        n = n + 1
    end

    return W, b
end

function learn_multiclass_logistic(lambda, m, eta, epochs)
    local W, b = minibatch_sgd(multiclass_logistic_gradient, multiclass_logistic_loss,
        lambda, m, eta, epochs)
    return W, b
end

function hinge_gradient(x, class, W, b)
    local y_hat = b:clone()
    if x:dim() ~= 0 then
        for i = 1, x:size(1) do
            y_hat:add(W:index(2, x:long()):sum(2))
        end
    end

    -- get the max non-true class
    local mask = torch.ones(nclasses):byte()
    mask[class] = 0
    local temp = y_hat:maskedSelect(mask)
    local _, cprime = temp:max(1)
    cprime = cprime[1]
    cprime = (cprime < class) and cprime or (cprime + 1)

    local grad = torch.zeros(nclasses)
    if y_hat[class] - y_hat[cprime] < 1 then
        grad[cprime] = 1
        grad[class] = -1
    end

    local grad_W = torch.zeros(nclasses, nfeatures)
    if x:dim() ~= 0 then
        for j = 1, x:size(1) do
            col = grad_W:select(2, x[j])
            col:add(grad)
        end
    end

    return grad_W, grad
end

function hinge_loss(x, class, W, b)
    local y_hat = b:clone()
    if x:dim() ~= 0 then
        for i = 1, x:size(1) do
            y_hat:add(W:index(2, x:long()):sum(2))
        end
    end

    -- get the max non-true class
    local mask = torch.ones(nclasses):byte()
    mask[class] = 0
    local temp = y_hat:maskedSelect(mask)
    local _, cprime = temp:max(1)
    cprime = cprime[1]
    cprime = (cprime < class) and cprime or (cprime + 1)

    return math.max(0, 1 - (y_hat[class] - y_hat[cprime]))
end

function learn_linear_svm(lambda, m, eta, epochs)
    local W, b = minibatch_sgd(hinge_gradient, hinge_loss,
        lambda, m, eta, epochs)
    return W, b
end


function main()
    -- Parse input params
    opt = cmd:parse(arg)
    local f = hdf5.open(opt.datafile, 'r')
    nclasses = f:read('nclasses'):all():long()[1]
    nfeatures = f:read('nfeatures'):all():long()[1]
    train_input = f:read("train_input"):all()
    train_output = f:read("train_output"):all()
    valid_input = f:read("valid_input"):all()
    valid_output = f:read("valid_output"):all()
    test_input = f:read("test_input"):all()
    classifier = opt.classifier
    alpha = opt.alpha
    lambda = opt.lambda
    m = opt.m
    eta = opt.eta
    epochs = opt.epochs


    local W = torch.DoubleTensor(nclasses, nfeatures)
    local b = torch.DoubleTensor(nclasses)

    print(f:read():all())

    -- Train.

    if classifier == "nb" then
        W, b = learn_naive_bayes(alpha)
    elseif classifier == "logistic" then
        W, b = learn_multiclass_logistic(lambda, m, eta, epochs)
    elseif classifier == "svm" then
        W, b = learn_linear_svm(lambda, m, eta, epochs)
    else
        print("Failed to recognize classifier ", classifier)
    end

    -- Test.
    eval_linear_model(W, b)

    local f_predictions = io.open("predictions.txt", "w")
    print_test_predictions(f_predictions, W, b)
    -- print_test_predictions(nil, W, b)
    f_predictions:close()
end

main()
