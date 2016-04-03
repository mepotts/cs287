-- Only requirement allowed
require("hdf5")
require("nn")
require("rnn")

cmd = torch.CmdLine()

-- Cmd Args
cmd:option('-datafile', 'PTB.hdf5', 'data file')
cmd:option('-classifier', 'mle_smooth', 'classifier to use')
cmd:option('-out', '', 'out file')
cmd:option('-use_dp', 1, 'use dp algorithm (slower)')

-- Only applicable for count-based and NN
cmd:option('-ngramsize', 7, 'ngram size')
cmd:option('-augment', 1, 'augment with next non-space chars')
-- Threshold for greedy algorithm
cmd:option('-thresh', 0.5, 'probability threshold')

-- Hyperparameters
-- Only applicable for count-based
cmd:option('-alpha', 0.01, 'smoothing value')

-- Only applicable for NN/RNN
cmd:option('-nembed', 20, 'embedding size')
cmd:option('-nhidden', 100, 'hidden layer size')
cmd:option('-eta', 0.01, 'learning rate')
cmd:option('-epochs', 30, 'number of epochs')

-- Only applicable for NN
cmd:option('-m', 32, 'batch size')
cmd:option('-normalize', 1, 'whether to normalize embeddings')
cmd:option('-downsample', 0, 'whether to downsample')

-- Only applicable for RNN
cmd:option('-nbatch', 20, 'batch size')
cmd:option('-nseq', 35, 'sequence size')

-- Constants
SPACE_CHAR = nil
START_CHAR = nil
END_CHAR = nil

function append_shift(row, c)
    return torch.cat(row, torch.Tensor({c})):narrow(1, 2, row:size(1))
end

function append_shiftnext(row, input, i, k)
    local ind = i
    for j = 1, k do
        while ind <= input:size(1) and input[ind] ~= END_CHAR and input[ind] == SPACE_CHAR do
            ind = ind + 1
        end
        local c = END_CHAR
        if ind <= input:size(1) and input[ind] ~= END_CHAR then
            c = input[ind]
        end
        row = append_shift(row, c)
    end
    return row
end

-- Converts 1D input into 2D windows with individual outputs
-- Each row is of size nfeatures = ngramsize - 1
-- Only for train/valid where the spaces are known
function get_rows(input)
    local row = nil
    local rows = {}
    local outs = {}
    for i = 1, input:size(1) do
        if row == nil then
            row = torch.Tensor(nfeatures):fill(START_CHAR)
        end
        local c = input[i]
        if c == END_CHAR then
            -- Don't try to predict END_CHAR
            row = nil
        else
            -- Assume no consecutive spaces, remove those from training
            if row[nfeatures] ~= SPACE_CHAR then
                local out = 1
                if c == SPACE_CHAR then
                    out = 2
                end
                if augment > 0 then
                    -- local nc = c
                    -- if c == SPACE_CHAR then
                    --     -- Guaranteed to be < input:size(1)
                    --     nc = input[i + 1]
                    --     assert(nc ~= SPACE_CHAR)
                    -- end
                    -- -- nc is first non-space character after prefix
                    -- local nrow = append_shift(row, nc)
                    local nrow = append_shiftnext(row, input, i, augment)
                    rows[#rows + 1] = nrow
                else
                    rows[#rows + 1] = row
                end
                outs[#outs + 1] = out
            end

            row = append_shift(row, c)
        end
    end
    local mat = torch.Tensor(#rows, nfeatures)
    for i = 1, #rows do
        mat[i] = rows[i]
    end
    return mat, torch.Tensor(outs)
end

-- Returns nchunk x nbatch x nseq
function get_chunks(input)
    print("nseq", nseq)
    print("nbatch", nbatch)
    local row = nil
    local rows = {}
    local outs = {}
    for i = 1, input:size(1) - nseq - 1, nseq do
        row = input:narrow(1, i, nseq)
        rows[#rows + 1] = row
        local targ = input:narrow(1, i + 1, nseq)
        -- Hack, maps to 1 if not space or 2 if space
        local out = (targ:eq(SPACE_CHAR):long() + targ:eq(END_CHAR):long()) + 1
        outs[#outs + 1] = out
    end
    local nchunk = torch.floor((#rows - nbatch) / nbatch)
    print("nchunk", nchunk)
    local batchrows = {}
    local batchouts = {}
    for i = 1, nchunk do
        local currows = torch.Tensor(nbatch, nseq)
        local curouts = torch.Tensor(nbatch, nseq)
        for j = 1, nbatch do
            local ind = (j-1) * nchunk + i
            currows[j] = rows[ind]
            curouts[j] = outs[ind]
        end
        batchrows[i] = currows
        batchouts[i] = curouts
    end
    return batchrows, batchouts
end

-- Takes a single sentence, with pref
-- fn is probability of space, compared against constant thresh
-- Returns predicted count
function predict_seg_greedy(input, fn)
    local row = torch.Tensor(nfeatures):fill(START_CHAR)
    local count = 0
    -- Don't predict END_CHAR
    for i = 1, input:size(1)-1 do
        local c = input[i]
        row = append_shift(row, c)
        local space = false
        if augment > 0 then
            -- local nc = input[i + 1]
            -- local nrow = append_shift(row, nc)
            local nrow = append_shiftnext(row, input, i+1, augment)
            if fn(nrow) > thresh then
                space = true
            end
        else
            if fn(row) > thresh then
                space = true
            end
        end

        if space then
            row = append_shift(row, SPACE_CHAR)
            count = count + 1
        end
    end
    return count
end

-- Takes a single sentence, with pref
-- fn is probability of space
-- Returns predicted count
function predict_seg_dp(input, fn)
    -- 2D dynamic programming
    -- state is (index, bitmask)
    -- bitmask is up to nfeatures-1 in length, records existence of past spaces
    -- bitmask & (1 << i) is true if the ith previous character has a space after it
    local dp = {}
    local par = {}
    for i = 1, input:size(1) do
        dp[i] = {}
        par[i] = {}
    end
    -- Mask to mask bitmask
    -- = (1 << (nfeatures-1)) - 1
    local mask = bit.lshift(1, nfeatures-1-augment) - 1
    -- Base case
    dp[1][0] = 0
    -- Don't predict END_CHAR
    for i = 1, input:size(1)-1 do
        for bitm, p in pairs(dp[i]) do
            -- Log probabilities
            local row = torch.Tensor(nfeatures):fill(START_CHAR)
            local bitind = 0
            -- Calculate probability that ith char has a space after it
            row[nfeatures] = input[i]
            local rowind = nfeatures - 1
            for j = i - 1, 1, -1 do
                if bit.band(bitm, bit.lshift(1, bitind)) > 0 then
                    row[rowind] = SPACE_CHAR
                    rowind = rowind - 1
                    if rowind == 0 then
                        break
                    end
                end
                row[rowind] = input[j]
                rowind = rowind - 1
                if rowind == 0 then
                    break
                end
                bitind = bitind + 1
            end
            local pred = nil
            if augment > 0 then
                -- local nc = input[i + 1]
                -- local nrow = append_shift(row, nc)
                local nrow = append_shiftnext(row, input, i+1, augment)
                pred = fn(nrow)
            else
                pred = fn(row)
            end

            local pspace = nil
            local pnospace = 0
            if pred > 0 then
                pspace = math.log(pred)
                pred = 1 - pred
                if pred > 0 then
                    pnospace = math.log(pred)
                else
                    pnospace = nil
                end
            end
            local nbit = bit.band(mask, bit.lshift(bitm, 1))
            if pnospace then
                local v = dp[i + 1][nbit]
                if not v or v < p + pnospace then
                    dp[i + 1][nbit] = p + pnospace
                    par[i + 1][nbit] = bitm
                end
            end
            if pspace then
                nbit = nbit + 1
                local v = dp[i + 1][nbit]
                if not v or v < p + pspace then
                    dp[i + 1][nbit] = p + pspace
                    par[i + 1][nbit] = bitm
                end
            end
        end
    end
    local i = input:size(1)
    local bestm = nil
    local bestp = nil
    for bitm, p in pairs(dp[i]) do
        if not bestp or bestp < p then
            bestm = bitm
            bestp = p
        end
    end
    local bitm = bestm
    local count = 0
    while bitm do
        count = count + bit.band(bitm, 1)
        bitm = par[i][bitm]
        i = i - 1
    end
    return count
end

-- Takes a single sentence, with pref
-- Runs lstm sequentially on input, returns log probability
-- Returns predicted count
function predict_seg_rnn(input, lstm)
    local count = 0
    -- Don't predict END_CHAR
    for i = 1, input:size(1)-1 do
        local c = input[i]
        local space = false
        local res = lstm:forward(torch.Tensor{{c}})
        if math.exp(res[1][1][2]) > thresh then
            space = true
        end

        if space then
            lstm:forward(torch.Tensor{{SPACE_CHAR}})
            count = count + 1
        end
    end
    -- Add last character, and then the END_CHAR
    lstm:forward(torch.Tensor{{input[input:size(1)]}})
    lstm:forward(torch.Tensor{{END_CHAR}})
    return count
end

-- Takes 1D input delimited by END_CHAR
-- Ignores spaces and calls predict fn on each sentence
-- Returns tensor of all counts
function predict_seg(input, fn)
    local sentence = {}
    local counts = {}
    for i = 1, input:size(1) do
        local c = input[i]
        if c == END_CHAR then
            local count = fn(torch.Tensor(sentence))
            sentence = {}
            counts[#counts + 1] = count
        elseif c ~= SPACE_CHAR then
            sentence[#sentence + 1] = c
        end
    end
    -- TODO: For some reason certain models tend to underestimate by 1
    -- return torch.Tensor(counts):long() + 1
    return torch.Tensor(counts):long()
end

function eval_seg(input, output, fn)
    local predict = predict_seg(input, fn)
    local nsegs = predict:size(1)
    print("nsegs", nsegs)
    print("expected", output:size(1))
    print(predict:narrow(1, 1, 10))
    print(output:narrow(1, 1, 10))
    local diff = predict - output
    print("average diff", diff:sum() / nsegs)
    local mse = math.pow(diff:float():norm(), 2) / nsegs
    print("mse", mse)
end

function print_seg(f, input, fn)
    local predict = predict_seg(input, fn)
    local nsegs = predict:size(1)
    print("nsegs", nsegs)

    local f = f or io.stdout
    f:write("ID,Count\n")
    for i = 1, nsegs do
        f:write(i, ",", predict[i], "\n")
    end
end


-- BEGIN code copied from HW3
-- Nested lookup tables for count-based model
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
            inc_multi(totals, sub, 1, nclasses * alpha)
        end
    end
    -- print_table(freqs)
    return freqs, totals, uniqs
end

emptytable = {}
function predict_mle(row, out, freqs, totals)
    local entry = get_multi(freqs, row) or emptytable
    local pred = entry[out] or alpha
    local all = get_multi(totals, row) or nclasses * alpha
    -- print(pred, all)
    return pred / all
end

function predict_smooth(row, out, freqs, totals, uniqs)
    local entry = get_multi(freqs, row) or emptytable
    local pred = entry[out] or alpha
    local all = get_multi(totals, row) or nclasses * alpha
    -- print(pred, all)
    local mle = pred / all
    if row:dim() > 0 and row:size(1) > 1 then
        local uniq = get_multi(uniqs, row)
        local ret = predict_smooth(row:narrow(1, 2, row:size(1)-1), out, freqs, totals, uniqs)
        if uniq then
            lambda = 1 - uniq / (uniq + all)
            return lambda * mle + (1 - lambda) * ret
        else
            -- Did not appear at all, use recurrowsrent value
            return ret
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


-- Functions for training NLM
function predict_mlp(input, output, mlp)
    local preds = mlp:forward(input)
    return math.exp(preds[output])
end


function predict_mlp_batched(input, options, mlp)
    local preds = mlp:forward(input)
    return preds:index(1, options):exp()
end


function getmlp()
    local mlp = nn.Sequential()

    -- words features
    local wordlookup = nn.LookupTable(nwords, nembed)
    mlp:add(wordlookup)
    mlp:add(nn.View(nfeatures*nembed))

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

    return err
end


function minibatch_sgd(train_input, train_output, valid_input, valid_output,
                       mlp, criterion, embedlayer)
    print("SGD m:", m)
    print("eta:", eta)
    print("epochs:", epochs)
    print("normalize:", normalize)
    print("downsample:", downsample)
    for i = 1, epochs do
        print("Iteration:", i)
        print("Time:", os.clock())

        -- Normalize to regularize at beginning of each epoch
        if normalize > 0 then
            -- Renormalizes across the second dimension
            embedlayer.weight:renorm(2, 1, 1)
        end

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
        print("Loss", total_err / perm:size(1))

        local fn = function(x, y) return predict_mlp(x, y, mlp) end
        -- Downsample when estimating perplexity
        print("Train")
        eval_predictor(train_input, train_output, fn, 100000)
        print("Valid")
        eval_predictor(valid_input, valid_output, fn, 100000)
    end
end


function learn_neural_network1(train_input, train_output, valid_input, valid_output)
    local mlp, embed, out_dim = getmlp(false)

    mlp:add(nn.Linear(out_dim, nhidden))
    mlp:add(nn.Tanh())
    -- Use nclasses here instead of nwords
    mlp:add(nn.Linear(nhidden, nclasses))
    mlp:add(nn.LogSoftMax())
    -- Don't average over batch size
    local criterion = nn.ClassNLLCriterion(nil, false)

    minibatch_sgd(train_input, train_output, valid_input, valid_output,
                  mlp, criterion, embed)
    return mlp
end
-- END code copied from HW3

function eval_rnn(input, output, lstm)
    local sum = 0
    for j = 1, #input do
        local data = input[j]:t()
        local out = output[j]:t()
        local preds = lstm:forward(data)
        for s = 1, nseq do
            for k = 1, nbatch do
                t = preds[s][k][out[s][k]]
                sum = sum + t
            end
        end
    end
    sum = -sum / (#input * nseq * nbatch)
    print("Perplexity", math.exp(sum))
end

function learn_rnn(train_input, train_output, valid_input, valid_output)
    local LT = nn.LookupTable(nwords, nembed)

    -- Using a Sequencer, let's make an LSTM that consumes a sequence of song-word embeddings
    local lstm = nn.Sequential()
    lstm:add(LT) -- for a single sequence, will return a sequence-length x embedDim tensor
    lstm:add(nn.SplitTable(1)) -- splits tensor into a sequence-length table containing vectors of size embedDim
    local rnnlayer = nn.LSTM(nembed, nhidden)
    lstm:add(nn.Sequencer(rnnlayer))
    lstm:add(nn.Sequencer(nn.Linear(nhidden, 2)))
    lstm:add(nn.Sequencer(nn.LogSoftMax())) -- map last state to a score for classification
    lstm:remember('both')

    local params, grad_params = lstm:getParameters()
    print('params', params:size())
    params:uniform(-0.05, 0.05)

    -- Don't average over batch size
    local crit = nn.SequencerCriterion(nn.ClassNLLCriterion(nil, false))

    local nchunk = #train_input
    print("eta:", eta)
    print("epochs:", epochs)
    for i = 1, epochs do
        print("Iteration:", i)
        print("Time:", os.clock())

        rnnlayer:forget()
        local total_loss = 0
        local total_count = 0
        for j = 1, nchunk do
            local data = train_input[j]:t()
            local out = train_output[j]:t()
            local preds = lstm:forward(data)
            -- print("out", out)
            -- t = torch.Tensor(nseq, nbatch)
            -- for i, v in ipairs(preds) do
            --     t[i] = v:select(2, 2)
            -- end
            -- print("preds", t)
            local loss = crit:forward(preds, out)
            -- print("loss", loss)
            total_loss = total_loss + loss
            local dLdPreds = crit:backward(preds, out)
            lstm:backward(data, dLdPreds)
            local norm = grad_params:norm()
            -- print("norm", norm)
            if norm > 5 then
                grad_params:div(norm):mul(5)
            end
            lstm:updateParameters(eta)
        end
        print("Loss", total_loss / (nchunk * nbatch * nseq))

        print("Valid")
        rnnlayer:forget()
        eval_rnn(valid_input, valid_output, lstm)
    end

    return lstm, rnnlayer
end


function main()
    -- Parse input params
    opt = cmd:parse(arg)
    local f = hdf5.open(opt.datafile, 'r')

    local classifier = opt.classifier
    print("Classifier", classifier)
    local outfile = opt.out
    print("Outfile", outfile)

    local train_input = f:read("train_input"):all():long()
    local valid_input = f:read("valid_input"):all():long()
    local valid_seg_input = f:read("valid_seg_input"):all():long()
    local valid_seg_output = f:read("valid_seg_output"):all():long()
    local test_seg_input = f:read("test_seg_input"):all():long()

    print("train_input", train_input:size(1))
    print("valid_input", valid_input:size(1))
    print("valid_seg_input", valid_seg_input:size(1))
    print("valid_seg_output", valid_seg_output:size(1))
    print("test_seg_input", test_seg_input:size(1))

    SPACE_CHAR = f:read('space_char'):all():long()[1]
    START_CHAR = f:read('start_char'):all():long()[1]
    END_CHAR = f:read('end_char'):all():long()[1]

    print("SPACE_CHAR", SPACE_CHAR)
    print("START_CHAR", START_CHAR)
    print("END_CHAR", END_CHAR)

    nwords = f:read('nwords'):all():long()[1]
    -- Either nospace/space
    nclasses = 2
    ngramsize = opt.ngramsize
    augment = opt.augment
    nfeatures = ngramsize - 1
    if augment > 0 then
        nfeatures = nfeatures + augment
    end

    print("nwords", nwords)
    print("nclasses", nclasses)
    print("ngramsize", ngramsize)
    print("nfeatures", nfeatures)

    alpha = opt.alpha

    thresh = opt.thresh
    print("threshold", thresh)

    nembed = opt.nembed
    nhidden = opt.nhidden
    print("nembed", nembed)
    print("nhidden", nhidden)

    m = opt.m
    epochs = opt.epochs
    eta = opt.eta
    normalize = opt.normalize
    downsample = opt.downsample

    -- Only for rnn
    nseq = opt.nseq
    nbatch = opt.nbatch

    use_dp = opt.use_dp

    -- Train.
    if classifier == "mle" or classifier == "mle_smooth" or classifier == "nn" then
        print("Converting:")
        train_rows, train_out = get_rows(train_input)
        valid_rows, valid_out = get_rows(valid_input)
        print("train_rows", train_rows:size(1))
        print("valid_rows", valid_rows:size(1))
    elseif classifier == "rnn" then
        train_rows, train_out = get_chunks(train_input)
        valid_rows, valid_out = get_chunks(valid_input)
    end

    print("Init time:", os.clock())
    print("Training models:")
    if classifier == "mle" or classifier == "mle_smooth" then
        freqs, totals, uniqs = train_mle(train_rows, train_out)
    elseif classifier == "nn" then
        mlp = learn_neural_network1(train_rows, train_out, valid_rows, valid_out)
    elseif classifier == "rnn" then
        lstm, rnnlayer = learn_rnn(train_rows, train_out, valid_rows, valid_out)
    end

    print("Training time:", os.clock())
    print("Testing models:")

    if classifier == "rnn" then
        print("Train")
        rnnlayer:forget()
        eval_rnn(train_rows, train_out, lstm)
        print("Valid")
        rnnlayer:forget()
        eval_rnn(valid_rows, valid_out, lstm)

        print("Greedy RNN")
        rnnfn = function(input) return predict_seg_rnn(input, lstm) end
        rnnlayer:forget()
        eval_seg(valid_seg_input, valid_seg_output, rnnfn)
    else
        local fn = nil
        if classifier == "mle"  then
            print("-- Laplacian -- ")
            fn = function(x, y) return predict_mle(x, y, freqs, totals) end

        elseif classifier == "mle_smooth" then
            print("-- Witten-Bell smoothing -- ")
            fn = function(x, y) return predict_smooth(x, y, freqs, totals, uniqs) end

        elseif classifier == "nn" then
            print("-- Neural language model -- ")
            fn = function(x, y) return predict_mlp(x, y, mlp) end
        end

        if not fn then
            print("Invalid classifier", classifier)
            os.exit(1)
        end

        -- Compute perplexities
        print("Train")
        eval_predictor(train_rows, train_out, fn)
        print("Valid")
        eval_predictor(valid_rows, valid_out, fn)

        -- Probability of a space
        local fnspace = function(x) return fn(x, 2) end

        print("Greedy")
        greedyfn = function(input) return predict_seg_greedy(input, fnspace) end
        eval_seg(valid_seg_input, valid_seg_output, greedyfn)

        print("Using DP?", use_dp)
        if use_dp > 0 then
            print("DP")
            dpfn = function(input) return predict_seg_dp(input, fnspace) end
            eval_seg(valid_seg_input, valid_seg_output, dpfn)
        end
    end

    print("Testing time:", os.clock())

    if outfile and outfile:len() > 0 then
        print("Writing output:")
        local f_preds = io.open(outfile, "w")
        local predfn = greedyfn
        if use_dp > 0 then
            predfn = dpfn
        end
        if classifier == "rnn" then
            predfn = rnnfn
            rnnlayer:forget()
        end
        print_seg(f_preds, test_seg_input, predfn)
        print("Predicting time:", os.clock())
    end

    print("Done!")
end

main()
