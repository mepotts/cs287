-- Only requirements allowed
require("hdf5")
require("nn")
require("optim")

cmd = torch.CmdLine()

-- Cmd Args
cmd:option('-datafile', 'PTB.hdf5', 'data file')
cmd:option('-classifier', 'nb', 'classifier to use')
cmd:option('-out', '', 'print to file')

-- Hyperparameters
-- ...
cmd:option('-alpha', 0, 'smoothing value')
cmd:option('-lambda', 0, 'L2 regularization value')
cmd:option('-sample', 0, 'Downsample')
cmd:option('-m', 32, 'size of minibatch for gradient descent')
cmd:option('-eta', 0.005, 'learning rate')
cmd:option('-epochs', 20, 'number of epochs for gradient descent')
cmd:option('-zeroembed', 0, 'zero out embedding gradients')


function transform_words(X_w)
    for i = 1, X_w:size(1) do
        X_w[i]:long():add(torch.range(0, (nfeatures-1)*nwords, nwords):long())
    end
end


function transform_caps(X_c)
    for i = 1, X_c:size(1) do
        X_c[i]:long():add(torch.range(0, (nfeatures-1)*ncaps, ncaps):long())
    end
end


function predict_class_linear(x_w, x_c, W_w, W_c, b)
    local y_hat = b:clone()
    
    y_hat:add(W_w:index(2, x_w:long()):sum(2))
    y_hat:add(W_c:index(2, x_c:long()):sum(2))
    
    local _, class = y_hat:max(1) -- get the argmax
    
    return class[1]
end


function eval_linear_model(W_w, W_c, b)
    local ncorrect = 0

    for i = 1, valid_words:size(1) do
        local class = predict_class_linear(valid_words[i], valid_caps[i], W_w, W_c, b)

        if class == valid_output[i] then
            ncorrect = ncorrect + 1
        end
    end

    print("# correct: ", ncorrect)
    print("# total: ", valid_words:size(1))
    print("Accuracy: ", ncorrect / valid_words:size(1), "\n")
end

function predict_class_mlp(x, mlp)
    local preds = mlp:forward(x)
    
    local _, class = preds:max(1) -- get the argmax
    
    return class[1]
end


function eval_mlp(mlp)
    local ncorrect = 0

    for i = 1, valid_words:size(1) do
        local x = {valid_words[i], valid_caps[i]}
        local class = predict_class_mlp(x, mlp)

        if class == valid_output[i] then
            ncorrect = ncorrect + 1
        end
    end

    print("# correct: ", ncorrect)
    print("# total: ", valid_words:size(1))
    print("Accuracy: ", ncorrect / valid_words:size(1), "\n")
end

function predict_test_classes_linear(W_w, W_c, b)
    local predictions = torch.Tensor(test_words:size(1))

    for i = 1, test_words:size(1) do
        local x_w = test_words[i]
        local x_c = test_caps[i]
        local class = predict_class_linear(x_w, x_c, W_w, W_c, b)

        predictions[i] = class
    end

    return predictions
end

function print_test_predictions_linear(f, W_w, W_c, b)
    local f = f or io.stdout
    f:write("ID", ",", "Class", "\n")

    local predictions = predict_test_classes_linear(W_w, W_c, b)
    for i = 1, predictions:size(1) do
        f:write(i, ",", predictions[i], '\n')
    end
end

function predict_test_classes_mlp(mlp)
    local predictions = torch.Tensor(test_words:size(1))

    for i = 1, test_words:size(1) do
        local x_w = test_words[i]
        local x_c = test_caps[i]
        local x = {x_w, x_c}
        local class = predict_class_mlp(x, mlp)

        predictions[i] = class
    end

    return predictions
end

function print_test_predictions_mlp(f, mlp)
    local f = f or io.stdout
    f:write("ID", ",", "Class", "\n")

    local predictions = predict_test_classes_mlp(mlp)
    for i = 1, predictions:size(1) do
        f:write(i, ",", predictions[i], '\n')
    end
end


function learn_naive_bayes(alpha)
    local alpha = alpha or 0

    -- Compute the prior
    print("Prior alpha:", alpha)
    local prior = torch.ones(nclasses):mul(alpha)
    for i = 1, train_output:size(1) do
        prior[train_output[i]] = prior[train_output[i]] + 1
    end
    prior:div(train_output:size(1))
    local b = torch.log(prior)
    
    -- Construct count matrix
    print("Count matrix")
    local F_w = torch.ones(nclasses, nfeatures*nwords):mul(alpha)
    local F_c = torch.ones(nclasses, nfeatures*ncaps):mul(alpha)
    for i = 1, train_words:size(1) do
        local class = train_output[i]
        local x_w = train_words[i]
        local x_c = train_caps[i]
        
        for j = 1, nfeatures do
            local feature_w = x_w[j]
            F_w[class][feature_w] = F_w[class][feature_w] + 1
            local feature_c = x_c[j]
            F_c[class][feature_c] = F_c[class][feature_c] + 1
        end
    end
    
    -- Compute the posterior by normalizing count matrix
    print("Posterior")
    for i = 1, nclasses do
        for j = 1, nfeatures do
            local win_w = F_w:sub(i, i, (j-1)*nwords + 1, (j-1)*nwords + nwords)
            if torch.sum(win_w) ~= 0 then
                win_w:div(torch.sum(win_w))
            end
            local win_c = F_c:sub(i, i, (j-1)*ncaps + 1, (j-1)*ncaps + ncaps)
            if torch.sum(win_c) ~= 0 then
                win_c:div(torch.sum(win_c))
            end
        end
    end
    
    return torch.log(F_w), torch.log(F_c), b
    
    -- local W = torch.cat(torch.log(F_w), torch.log(F_c), 2)

    -- return W, b
end


function batch_grad_update(mlp, criterion, x, y, learning_rate, embedlayer)
    mlp:zeroGradParameters()
    local pred = mlp:forward(x)
    local err = criterion:forward(pred, y)
    local t = criterion:backward(pred, y)
    mlp:backward(x, t)
    if zeroembed then
        embedlayer.gradWeight:zero()
    end
    mlp:updateParameters(learning_rate)
    return err
end


function minibatch_sgd(mlp, criterion, lambda, m, eta, epochs, embedlayer)
    print("SGD m:", m)
    print("eta:", eta)
    print("zeroembed:", zeroembed)
    print("epochs:", epochs)
    print("sample:", downsample)
    for i = 1, epochs do
        print("Iteration:", i)
        print("Time:", os.clock())
        local perm = torch.randperm(train_words:size(1))

        -- Downsample for faster results
        if downsample then
            perm = perm:narrow(1, 1, downsample)
        end
        
        local total_err = 0
        for j = 1, perm:size(1) - m, m do
            local sample = perm:narrow(1, j, m):long()
            local x1 = train_words:index(1, sample)
            local x2 = train_caps:index(1, sample)
            local x = {x1, x2}
            local y = train_output:index(1, sample)
            
            local err = batch_grad_update(mlp, criterion, x, y, eta, embedlayer)
            total_err = total_err + err
        end
        print("Loss", total_err/train_words:size(1))
        eval_mlp(mlp)
        prlong()
    end
end


function getmlp(use_embedding)
    local mlp = nn.Sequential()

    local parallel = nn.ParallelTable()
    -- words features
    local wordtable = nn.Sequential()
    local wordlookup = nn.LookupTable(nfeatures*nwords, nembed)
    if use_embedding then
        for i = 1, nfeatures do
            for j = 1, nwords do
                wordlookup.weight[(i-1)*nwords + j] = embeddings[j]
            end
        end
    end
    wordtable:add(wordlookup)
    wordtable:add(nn.View(nfeatures*nembed))
    parallel:add(wordtable)

    -- caps features
    local captable = nn.Sequential()
    captable:add(nn.LookupTable(nfeatures*ncaps, nembed_caps))
    local caplookup = nn.LookupTable(nfeatures*nwords, nembed)
    captable:add(nn.View(nfeatures*nembed_caps))
    parallel:add(captable)

    mlp:add(parallel)
    mlp:add(nn.JoinTable(1, 1))
    return mlp, wordlookup, nfeatures * (nembed + nembed_caps)
end


function learn_multiclass_logistic(lambda, m, eta, epochs)
    local mlp, embed, out_dim = getmlp(false)

    mlp:add(nn.Linear(out_dim, nclasses))
    mlp:add(nn.LogSoftMax())
    local criterion = nn.ClassNLLCriterion()
    criterion.sizeAverage = false
    
    minibatch_sgd(mlp, criterion, lambda, m, eta, epochs, embed)
    return mlp
end


function learn_neural_network1(lambda, m, eta, epochs)
    local mlp, embed, out_dim = getmlp(false)

    mlp:add(nn.Linear(out_dim, nhidden))
    mlp:add(nn.HardTanh())
    mlp:add(nn.Linear(nhidden, nclasses))
    mlp:add(nn.LogSoftMax())
    local criterion = nn.ClassNLLCriterion()
    criterion.sizeAverage = false
    
    minibatch_sgd(mlp, criterion, lambda, m, eta, epochs, embed)
    return mlp
end


function learn_neural_network2(lambda, m, eta, epochs)
    local mlp, embed, out_dim = getmlp(true)

    mlp:add(nn.Linear(out_dim, nhidden))
    mlp:add(nn.HardTanh())
    mlp:add(nn.Linear(nhidden, nclasses))
    mlp:add(nn.LogSoftMax())
    local criterion = nn.ClassNLLCriterion()
    criterion.sizeAverage = false
    
    minibatch_sgd(mlp, criterion, lambda, m, eta, epochs, embed)
    return mlp
end


function main() 
    -- Parse input params
    opt = cmd:parse(arg)
    local f = hdf5.open(opt.datafile, 'r')
    outfile = opt.out

    print("Reading")
    print("datafile", opt.datafile)

    train_words = f:read("train_input_word_windows"):all()
    train_caps = f:read("train_input_cap_windows"):all()
    train_output = f:read("train_output"):all()
    valid_words = f:read("valid_input_word_windows"):all()
    valid_caps = f:read("valid_input_cap_windows"):all()
    valid_output = f:read("valid_output"):all()
    test_words = f:read("test_input_word_windows"):all()
    test_caps = f:read("test_input_cap_windows"):all()
    embeddings = f:read("word_embeddings"):all()

    nclasses = f:read('nclasses'):all():long()[1]
    nwords = f:read('nwords'):all():long()[1]
    nfeatures = train_words:size(2)
    nembed = embeddings:size(2)
    ncaps = 4

    print("Transforming")

    transform_words(train_words)
    transform_caps(train_caps)
    transform_words(valid_words)
    transform_caps(valid_caps)
    transform_words(test_words)
    transform_caps(test_caps)

    nembed_caps = 5
    nhidden = 300
    classifier = opt.classifier
    alpha = opt.alpha
    lambda = opt.lambda
    downsample = opt.sample
    m = opt.m
    eta = opt.eta
    epochs = opt.epochs
    zeroembed = opt.zeroembed

    print("train_words", train_words:size(1))
    print("classifier", classifier)
    print("nclasses", nclasses)
    print("nwords", nwords)
    print("nfeatures", nfeatures)
    print("ncaps", ncaps)

    -- local W_w = torch.DoubleTensor(nclasses, nfeatures)
    -- local b = torch.DoubleTensor(nclasses)
    
    print("Time start:", os.clock())

    -- Train.
    if classifier == "nb" then
        W_w, W_c, b = learn_naive_bayes(alpha)
    elseif classifier == "logistic" then
        mlp = learn_multiclass_logistic(lambda, m, eta, epochs)
    elseif classifier == "nn1" then
        mlp = learn_neural_network1(lambda, m, eta, epochs)
    elseif classifier == "nn2" then
        mlp = learn_neural_network2(lambda, m, eta, epochs)
    end

    print("Time done:", os.clock())

    -- Test.
    if classifier == "nb" then
        eval_linear_model(W_w, W_c, b)
        if outfile then
            local f_predictions = io.open(outfile, "w")
            print_test_predictions_linear(f_predictions, W_w, W_c, b)
            f_predictions:close()
        end
    elseif classifier == "logistic" or classifier == "nn1" or classifier == "nn2" then
        eval_mlp(mlp)
        if outfile then
            local f_predictions = io.open(outfile, "w")
            print_test_predictions_mlp(f_predictions, mlp)
            f_predictions:close()
        end
    end
end

main()
