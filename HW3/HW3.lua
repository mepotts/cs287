-- Only requirements allowed
require("hdf5")
require("nn")
require("optim")

cmd = torch.CmdLine()

-- Cmd Args
cmd:option('-datafile', 'PTB.hdf5', 'data file')
cmd:option('-out', '', 'out file')
cmd:option('-embedout', '', 'embed out file')
cmd:option('-valid', 0, 'validate?')
cmd:option('-classifier', 'mle', 'classifier to use')

-- Hyperparameters
-- ...
cmd:option('-alpha', 0.005, 'smoothing value')
cmd:option('-eta', 0.005, 'learning rate')
cmd:option('-epochs', 20, 'number of epochs')
cmd:option('-normalize', 1, 'whether to normalize embeddings')
cmd:option('-m', 32, 'batch size')
cmd:option('-nembed', 60, 'embedding size')
cmd:option('-nhidden', 100, 'hidden layer size')
cmd:option('-downsample', 0, 'whether to downsample')
cmd:option('-k', 32, 'sampling size')

function transform_dummy(data)
    -- Adds an extra column as a base case for smoothing
    return data:cat(torch.Tensor(data:size(1), 1):zero():long())
end

function set_size(table, tensor)
    local size = tensor:size(1)
    local child = table[size]
    if not child then
        table[size] = {}
        child = table[size]
    end
    return child
end

function inc_multi(table, tensor, inc, default)
    local table = set_size(table, tensor)
    for i = 1, tensor:size(1) - 1 do
        local v = tensor[i]
        local child = table[v]
        if not child then
            table[v] = {}
            child = table[v]
        end
        table = child
    end
    v = tensor[tensor:size(1)]
    table[v] = (table[v] or default) + inc
end

function get_multi(table, tensor)
    local table = set_size(table, tensor)
    for i = 1, tensor:size(1) do
        local v = tensor[i]
        local child = table[v]
        if not child then
            return nil
        end
        table = child
    end
    return table
end

function set_multi(table, tensor, value)
    local table = set_size(table, tensor)
    for i = 1, tensor:size(1) - 1 do
        local v = tensor[i]
        local child = table[v]
        if not child then
            table[v] = {}
            child = table[v]
        end
        table = child
    end
    v = tensor[tensor:size(1)]
    table[v] = value
end

function get_table_multi(table, tensor)
    local table = set_size(table, tensor)
    for i = 1, tensor:size(1) do
        local v = tensor[i]
        local child = table[v]
        if not child then
            table[v] = {}
            child = table[v]
        end
        table = child
    end
    return table
end

function train_mle(input, output)
    print("alpha", alpha)
    local totals = {}
    local freqs = {}
    local uniqs = {}
    for i = 1, input:size(1) do
        local row = input[i]
        for j = 1, row:size(1) do
            local sub = row:narrow(1, j, row:size(1) + 1 - j)
            local out = output[i]
            local entry = get_table_multi(freqs, sub)
            if not entry[out] then
                inc_multi(uniqs, sub, 1, 0)
            end
            entry[out] = (entry[out] or alpha) + 1
            inc_multi(totals, sub, 1, nwords * alpha)
        end
    end
    -- print_table(freqs)
    return freqs, totals, uniqs
end

emptytable = {}
function predict_mle(row, out, freqs, totals)
    local entry = get_multi(freqs, row) or emptytable
    local pred = entry[out] or alpha
    local all = get_multi(totals, row) or nwords * alpha
    -- print(pred, all)
    return pred / all
end

function predict_smooth(row, out, freqs, totals, uniqs)
    local entry = get_multi(freqs, row) or emptytable
    local pred = entry[out] or alpha
    local all = get_multi(totals, row) or nwords * alpha
    -- print(pred, all)
    local mle = pred / all
    if row:dim() > 0 and row:size(1) > 1 then
        local uniq = get_multi(uniqs, row)
        if uniq then
            lambda = 1 - uniq / (uniq + all)
            local ret = predict_smooth(row:narrow(1, 2, row:size(1)-1), out, freqs, totals, uniqs)
            return lambda * mle + (1 - lambda) * ret
        end
    end
    return mle
end

function eval_predictor(input, output, fn, sample)
    local sum = 0
    if sample and sample > 0 and sample < input:size(1) then
        local perm = torch.randperm(input:size(1))
        perm = perm:narrow(1, 1, sample):long()
        input = input:index(1, perm)
        output = output:index(1, perm)
    end

    for i = 1, input:size(1) do
        local res = fn(input[i], output[i])
        sum = sum + math.log(res)
    end
    sum = -sum / input:size(1)

    for i = 1, 10 do
        local res = fn(input[i], output[i])
        print(math.log(res))
    end

    local perp = math.exp(sum)
    print("Perplexity", perp)
end

function print_predictions(f, input, options, fn)
    local f = f or io.stdout
    local noptions = options:size(2)
    f:write("ID")
    for i = 1, noptions do
        f:write(",Class", i)
    end
    f:write("\n")
    for i = 1, input:size(1) do
        local row = input[i]
        local cur = options[i]
        local sum = 0
        local preds = fn(row, cur)
        preds = preds / preds:sum()
        f:write(i)
        for j = 1, preds:size(1) do
            f:write(",", string.format("%.7f", preds[j]))
        end
        f:write("\n")
    end
end


function predict_mlp(input, output, mlp)
    local preds = mlp:forward(input)
    return math.exp(preds[output])
end


function predict_mlp_batched(input, options, mlp)
    local preds = mlp:forward(input)
    return preds:index(1, options):exp()
end


function predict_mlp_batched2(input, options, mlp, sublinear, linear)
    local n = options:size(1)
    sublinear.weight:narrow(1, 1, n):copy(linear.weight:index(1, options))
    sublinear.bias:narrow(1, 1, n):copy(linear.bias:index(1, options))
    local preds = mlp:forward(input)
    local res = preds:narrow(1, 1, n)
    -- Normalize by max before exp
    res = res - res:max()
    res = res:exp()
    return res / res:sum()
end


function batched(input, options, fn)
    local res = torch.Tensor(options:size(1))
    for i = 1, options:size(1) do
        res[i] = fn(input, options[i])
    end
    return res
end


function getmlp()
    local mlp = nn.Sequential()

    -- words features
    local wordlookup = nn.LookupTable(nwords, nembed)

    mlp:add(wordlookup)
    mlp:add(nn.View(nfeatures*nembed))
    if normalize > 0 then
        -- Renormalizes across the second dimension
        wordlookup.weight:renorm(2, 1, 1)
    end

    return mlp, wordlookup, nfeatures * nembed
end


function batch_grad_update(mlp, criterion, x, y, embedlayer)
    mlp:zeroGradParameters()
    local pred = mlp:forward(x)
    -- print(pred:size())
    local err = criterion:forward(pred, y)
    -- print(err)
    local t = criterion:backward(pred, y)
    mlp:backward(x, t)
    mlp:updateParameters(eta)

    if normalize > 0 then
        -- Renormalizes across the second dimension
        embedlayer.weight:renorm(2, 1, 1)
    end
    return err
end


function minibatch_sgd(mlp, criterion, embedlayer)
    print("SGD m:", m)
    print("eta:", eta)
    print("epochs:", epochs)
    print("normalize:", normalize)
    print("downsample:", downsample)
    for i = 1, epochs do
        print("Iteration:", i)
        print("Time:", os.clock())
        local perm = torch.randperm(train_input:size(1))

        -- Downsample for faster results
        if downsample > 0 then
            perm = perm:narrow(1, 1, downsample)
        end

        local total_err = 0
        for j = 1, perm:size(1) - m, m do
            local sample = perm:narrow(1, j, m):long()
            local x = train_input:index(1, sample)
            local y = train_output:index(1, sample)

            local err = batch_grad_update(mlp, criterion, x, y, embedlayer)
            total_err = total_err + err
        end
        print("Loss", total_err/perm:size(1))

        local fn = function(x, y) return predict_mlp(x, y, mlp) end
        -- Downsample when estimating perplexity
        print("Train")
        eval_predictor(train_input, train_output, fn, 100000)
        print("Valid")
        eval_predictor(valid_input, valid_output, fn, 100000)
    end
end


function learn_neural_network1()
    local mlp, embed, out_dim = getmlp(false)

    mlp:add(nn.Linear(out_dim, nhidden))
    mlp:add(nn.Tanh())
    mlp:add(nn.Linear(nhidden, nwords))
    mlp:add(nn.LogSoftMax())
    local criterion = nn.ClassNLLCriterion()
    criterion.sizeAverage = false

    minibatch_sgd(mlp, criterion, embed)
    return mlp
end


function predict_mlp2(input, output, mlp, sublinear, linear)
    sublinear.weight[1] = linear.weight[output]
    sublinear.bias[1] = linear.bias[output]
    local preds = mlp:forward(input)
    return math.exp(preds[1])
end


function sigmoid(t)
    return 1 / (1 + math.exp(-t))
end


function batch_grad_update2(mlp, x, y, embedlayer, sublinear, linear)
    mlp:zeroGradParameters()
    assert(m == x:size(1))
    assert(m == y:size(1))
    sampling = torch.multinomial(freq_array, k)
    assert(k == sampling:size(1))

    for i = 1, m do
        sublinear.weight[i] = linear.weight[y[i]]
        sublinear.bias[i] = linear.bias[y[i]]
    end
    for i = 1, k do
        sublinear.weight[i + m] = linear.weight[sampling[i]]
        sublinear.bias[i + m] = linear.bias[sampling[i]]
    end

    local pred = mlp:forward(x)
    local err = 0
    local t = torch.Tensor(m, m + k):zero()
    for i = 1, m do
        local vi = sigmoid(pred[i][i] - math.log(freq_array[y[i]]))
        err = err + math.log(vi)
        t[i][i] = 1 - vi
        for j = 1, k do
            local vj = sigmoid(pred[i][j + m] - math.log(freq_array[sampling[j]]))
            err = err + math.log(1 - vj)
            t[i][j + m] = -vj
        end
    end
    -- Calculated objective, loss is negative
    err = -err
    t = -t
    -- print(err)
    -- print(t)
    mlp:backward(x, t)
    mlp:updateParameters(eta)

    for i = 1, m do
        linear.weight[y[i]] = linear.weight[y[i]] - sublinear.gradWeight[i] * eta
        linear.bias[y[i]] = linear.bias[y[i]] - sublinear.gradBias[i] * eta
    end
    for i = 1, k do
        linear.weight[sampling[i]] = linear.weight[sampling[i]] - sublinear.gradWeight[i + m] * eta
        linear.bias[sampling[i]] = linear.bias[sampling[i]] - sublinear.gradBias[i + m] * eta
    end

    if normalize > 0 then
        -- Renormalizes across the second dimension
        embedlayer.weight:renorm(2, 1, 1)
    end
    return err
end


function minibatch_sgd2(mlp, embedlayer, sublinear, linear)
    print("SGD m:", m)
    print("eta:", eta)
    print("epochs:", epochs)
    print("normalize:", normalize)
    print("downsample:", downsample)
    for i = 1, epochs do
        print("Iteration:", i)
        print("Time:", os.clock())
        local perm = torch.randperm(train_input:size(1))

        -- Downsample for faster results
        if downsample > 0 then
            perm = perm:narrow(1, 1, downsample)
        end

        local total_err = 0
        for j = 1, perm:size(1) - m, m do
            local sample = perm:narrow(1, j, m):long()
            local x = train_input:index(1, sample)
            local y = train_output:index(1, sample)

            local err = batch_grad_update2(mlp, x, y, embedlayer, sublinear, linear)
            total_err = total_err + err
        end
        print("Loss", total_err/perm:size(1))

        local fn = function(x, y) return predict_mlp2(x, y, mlp, sublinear, linear) end
        -- Downsample when estimating perplexity
        print("Train")
        eval_predictor(train_input, train_output, fn, 100000)
        print("Valid")
        eval_predictor(valid_input, valid_output, fn, 100000)
    end
end


function learn_neural_network2()
    local mlp, embed, out_dim = getmlp(false)

    mlp:add(nn.Linear(out_dim, nhidden))
    mlp:add(nn.Tanh())
    local linear = nn.Linear(nhidden, nwords)
    -- Include m mini-batch and k counterexamples
    local sublinear = nn.Linear(nhidden, m + k)
    mlp:add(sublinear)
    -- No criterion - roll our own

    minibatch_sgd2(mlp, embed, sublinear, linear)

    mlp:remove()
    mlp:add(linear)
    mlp:add(nn.LogSoftMax())
    return mlp
end


function compute_freq(output)
    freq_array = torch.Tensor(nwords):zero()
    for i = 1, output:size(1) do
        local out = output[i]
        freq_array[out] = freq_array[out] + 1
    end
    freq_array[freq_array:eq(0)] = 1
    freq_array = freq_array / freq_array:sum()
end


function main()
    -- Parse input params
    opt = cmd:parse(arg)
    local f = hdf5.open(opt.datafile, 'r')
    classifier = opt.classifier
    print("Classifier", classifier)
    outfile = opt.out
    print("Outfile", outfile)
    embedout = opt.embedout
    print("Embed out", embedout)
    validate = opt.valid
    print("Validate", validate)

    print("Reading")
    print("datafile", opt.datafile)

    train_input = f:read("train_input"):all():long()
    train_output = f:read("train_output"):all():long()
    valid_input = f:read("valid_input"):all():long()
    valid_output = f:read("valid_output"):all():long()
    valid_blanks_input = f:read("valid_blanks_input"):all():long()
    valid_blanks_options = f:read("valid_blanks_options"):all():long()
    test_blanks_options = f:read("test_blanks_options"):all():long()
    test_blanks_input = f:read("test_blanks_input"):all():long()

    nwords = f:read('nwords'):all():long()[1]
    nfeatures = train_input:size(2)

    print("nwords", nwords)
    print("nfeatures", nfeatures)

    alpha = opt.alpha

    nembed = opt.nembed
    nhidden = opt.nhidden
    m = opt.m
    epochs = opt.epochs
    eta = opt.eta
    normalize = opt.normalize
    downsample = opt.downsample
    k = opt.k

    if classifier == "mle" then
        print("Transforming data:")
        train_input = transform_dummy(train_input)
        valid_input = transform_dummy(valid_input)
        valid_blanks_input = transform_dummy(valid_blanks_input)
        test_blanks_input = transform_dummy(test_blanks_input)
    end

    print("Computing frequencies:")
    compute_freq(train_output)

    print("Init time:", os.clock())
    print("Training models:")
    print("train dims:")
    print(train_input:size())

    if classifier == "mle" then
        freqs, totals, uniqs = train_mle(train_input, train_output)
        print("freqs[1][0]", freqs[1][0])
        print("uniqs[1][0]", uniqs[1][0])
    elseif classifier == "nn1" then
        mlp = learn_neural_network1()
    elseif classifier == "nn2" then
        mlp = learn_neural_network2()
    else
        print("Invalid classifier", classifier)
    end

    print("Training time:", os.clock())
    print("Testing models:")

    if classifier == "mle" then
        local fn = function(x, y) return predict_mle(x, y, freqs, totals) end
        print("Train")
        eval_predictor(train_input, train_output, fn)
        print("Valid")
        eval_predictor(valid_input, valid_output, fn)

        local fnsmooth = function(x, y) return predict_smooth(x, y, freqs, totals, uniqs) end
        print("Train")
        eval_predictor(train_input, train_output, fnsmooth)
        print("Valid")
        eval_predictor(valid_input, valid_output, fnsmooth)
    elseif classifier == "nn1" or classifier == "nn2" then
        local fn = function(x, y) return predict_mlp(x, y, mlp) end
        print("Train")
        eval_predictor(train_input, train_output, fn)
        print("Valid")
        eval_predictor(valid_input, valid_output, fn)
    else
        print("Invalid classifier", classifier)
    end

    print("Testing time:", os.clock())

    if outfile and outfile:len() > 0 then
        print("Writing output:")
        local input
        local options
        if validate > 0 then
            input = valid_blanks_input
            options = valid_blanks_options
        else
            input = test_blanks_input
            options = test_blanks_options
        end
        local f_predictions = io.open(outfile, "w")
        if classifier == "mle" then
            local fnsmooth = function(x, y) return predict_smooth(x, y, freqs, totals, uniqs) end
            local fn = function(x, options) return batched(x, options, fnsmooth) end
            print_predictions(f_predictions, input, options, batched)
        elseif classifier == "nn1" or classifier == "nn2" then
            local fn = function(x, options) return predict_mlp_batched(x, options, mlp) end
            print_predictions(f_predictions, input, options, fn)
        end
    end

    print("Predicting time:", os.clock())

    if embedout and embedout:len() > 0 then
        print("Writing embed:")
        local embedfile = hdf5.open(embedout, 'w')
        if classifier == "nn1" or classifier == "nn2" then
            embedfile:write('embedding', mlp:get(1).weight)
        end
    end
end

main()
