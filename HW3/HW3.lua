-- Only requirements allowed
require("hdf5")
require("nn")
require("optim")

cmd = torch.CmdLine()

-- Cmd Args
cmd:option('-out', '', 'out file')
cmd:option('-valid', 0, 'validate?')
cmd:option('-datafile', 'PTB.hdf5', 'data file')
cmd:option('-classifier', 'mle', 'classifier to use')

-- Hyperparameters
-- ...
cmd:option('-alpha', 0.005, 'smoothing value')

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

function eval_predictor(input, output, fn)
    local sum = 0
    for i = 1, input:size(1) do
        local res = fn(input[i], output[i])
        sum = sum + math.log(res)
    end
    sum = -sum / input:size(1)
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
        local preds = torch.Tensor(cur:size(1))
        for j = 1, cur:size(1) do
            local out = cur[j]
            local res = fn(row, out)
            preds[j] = res
            sum = sum + res
        end
        preds = preds / sum
        f:write(i)
        for j = 1, preds:size(1) do
            f:write(",", string.format("%.7f", preds[j]))
        end
        f:write("\n")
    end
end

function main()
    -- Parse input params
    opt = cmd:parse(arg)
    local f = hdf5.open(opt.datafile, 'r')
    classifier = opt.classifier
    print("Classifier", classifier)
    outfile = opt.out
    print("Outfile", outfile)
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

    print("alpha", alpha)

    if classifier == "mle" then
        print("Transforming data:")
        train_input = transform_dummy(train_input)
        valid_input = transform_dummy(valid_input)
        valid_blanks_input = transform_dummy(valid_blanks_input)
        test_blanks_input = transform_dummy(test_blanks_input)
    end

    print("Training models:")
    print("train dims:")
    print(train_input:size())

    if classifier == "mle" then
        freqs, totals, uniqs = train_mle(train_input, train_output)
        print("freqs[1][0]", freqs[1][0])
        print("uniqs[1][0]", uniqs[1][0])
    else
        print("Invalid classifier", classifier)
    end

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
    else
        print("Invalid classifier", classifier)
    end

    if outfile:len() > 0 then
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
        if classifier == "mle" then
            local fnsmooth = function(x, y) return predict_smooth(x, y, freqs, totals, uniqs) end
            local f_predictions = io.open(outfile, "w")
            print_predictions(f_predictions, input, options, fnsmooth)
        end
    end
end

main()