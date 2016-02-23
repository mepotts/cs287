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
cmd:option('-lambda', 0, 'L2 regularization value')
cmd:option('-m', 32, 'size of minibatch for gradient descent')
cmd:option('-eta', 0.01, 'learning rate')
cmd:option('-epochs', 20, 'number of epochs for gradient descent')


function predict_class_linear(x_w, x_c, W_w, W_c, b)
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
        local class = predict_class_mlp(valid_words[i], mlp)

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
        local class = predict_class_mlp(x_w, mlp)

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

function batch_grad_update(mlp, criterion, x, y, learning_rate)
	mlp:zeroGradParameters()
	local pred = mlp:forward(x)
	local err = criterion:forward(pred, y)
	local t = criterion:backward(pred, y)
	mlp:backward(x, t)
	mlp:updateParameters(learning_rate)
	return err
end



function minibatch_sgd(mlp, criterion, lambda, m, eta, epochs)
	for i = 1, epochs do
		print("Iteration", i)
		local perm = torch.randperm(train_words:size(1))
		
		local total_err = 0
		for j = 1, perm:size(1), m do
			local sample = perm:narrow(1, j, torch.min(torch.Tensor({m, perm:size(1)-j+1}))):long()
			local x = train_words:index(1, sample)
			local y = train_output:index(1, sample)
			
			local err = batch_grad_update(mlp, criterion, x, y, eta)
			total_err = total_err + err
		end
		print("Loss", total_err/train_words:size(1))
		eval_mlp(mlp)
		print()
	end
end


function learn_multiclass_logistic(lambda, m, eta, epochs)
	local mlp = nn.Sequential()
	mlp:add(nn.LookupTable(nfeatures*nwords, nclasses))
	mlp:add(nn.Sum(1))
	mlp:add(nn.Add(nclasses))
	mlp:add(nn.LogSoftMax())
	local criterion = nn.ClassNLLCriterion()
	criterion.sizeAverage = false
	
	minibatch_sgd(mlp, criterion, lambda, m, eta, epochs)
	return mlp
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
	lambda = opt.lambda
	m = opt.m
	eta = opt.eta
	epochs = opt.epochs


	-- local W_w = torch.DoubleTensor(nclasses, nfeatures)
	-- local b = torch.DoubleTensor(nclasses)

	-- Train.
	if classifier == "nb" then
	    W_w, W_c, b = learn_naive_bayes(alpha)
	elseif classifier == "logistic" then
		mlp = learn_multiclass_logistic(lambda, m, eta, epochs)
	end


	-- Test.
	if classifier == "nb" then
		eval_linear_model(W_w, W_c, b)
	    local f_predictions = io.open("predictions.txt", "w")
	    print_test_predictions_linear(f_predictions, W_w, W_c, b)
	    f_predictions:close()
	elseif classifier == "logistic" then
		eval_mlp(mlp)
	    local f_predictions = io.open("predictions.txt", "w")
	    print_test_predictions_mlp(f_predictions, mlp)
	    f_predictions:close()
	end
end

main()
