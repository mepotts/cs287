-- Only requirements allowed
require("hdf5")
require("nn")
require("optim")

cmd = torch.CmdLine()

-- Cmd Args
cmd:option('-datafile', '', 'data file')
cmd:option('-classifier', 'nb', 'classifier to use')

-- Hyperparameters
-- ...
cmd:option('-alpha', 0, 'smoothing value')


function predict_class(x_w, x_c, W_w, W_c, b)
    local y_hat = b:clone()
    
    for i = 1, x_w:size(1) do
        y_hat:add(W_w:index(2, torch.range(0, (nfeatures-1)*nwords, nwords):long():add(x_w:long())):sum(2))
        y_hat:add(W_c:index(2, torch.range(0, (nfeatures-1)*ncaps, ncaps):long():add(x_c:long())):sum(2))
    end
    
    local _, class = y_hat:max(1) -- get the argmax
    
    return class[1]
end


function eval_linear_model(W_w, W_c, b)
    local ncorrect = 0

    for i = 1, valid_words:size(1) do
        local class = predict_class(valid_words[i], valid_caps[i], W_w, W_c, b)

        if class == valid_output[i] then
            ncorrect = ncorrect + 1
        end
    end

    print("# correct: ", ncorrect)
    print("# total: ", valid_words:size(1))
    print("Accuracy: ", ncorrect / valid_words:size(1), "\n")
end


function learn_naive_bayes(alpha)
    local alpha = alpha or 0

    -- Compute the prior
    local prior = torch.ones(nclasses):mul(alpha)
    for i = 1, train_output:size(1) do
        prior[train_output[i]] = prior[train_output[i]] + 1
    end
    prior:div(train_output:size(1))
    local b = torch.log(prior)
    
    -- Construct count matrix
    local F_w = torch.ones(nclasses, nfeatures*nwords):mul(alpha)
    local F_c = torch.ones(nclasses, nfeatures*ncaps):mul(alpha)
    for i = 1, train_words:size(1) do
        local class = train_output[i]
        local x_w = train_words[i]
        local x_c = train_caps[i]
        
        for j = 1, nfeatures do
            local feature_w = x_w[j]
            F_w[class][(j-1)*nwords+feature_w] = F_w[class][(j-1)*nwords+feature_w] + 1
            local feature_c = x_c[j]
            F_c[class][(j-1)*ncaps+feature_c] = F_c[class][(j-1)*ncaps+feature_c] + 1
        end
    end
    
    -- Compute the posterior by normalizing count matrix
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



function main() 
	-- Parse input params
	opt = cmd:parse(arg)
	local f = hdf5.open(opt.datafile, 'r')
	train_words = f:read("train_input_word_windows"):all()
	train_caps = f:read("train_input_cap_windows"):all()
	train_output = f:read("train_output"):all()
	valid_words = f:read("valid_input_word_windows"):all()
	valid_caps = f:read("valid_input_cap_windows"):all()
	valid_output = f:read("valid_output"):all()
	test_words = f:read("test_input_word_windows"):all()
	test_caps = f:read("test_input_cap_windows"):all()
	nclasses = f:read('nclasses'):all():long()[1]
	nwords = f:read('nwords'):all():long()[1]
	nfeatures = train_words:size(2)
	ncaps = 4
    classifier = opt.classifier
	
	-- local W_w = torch.DoubleTensor(nclasses, nfeatures)
	-- local b = torch.DoubleTensor(nclasses)
	
	-- Train.
	if classifier == "nb" then
	    W_w, W_c, b = learn_naive_bayes(alpha)
	end
	
	
	-- Test.
	eval_linear_model(W_w, W_c, b)
end

main()
