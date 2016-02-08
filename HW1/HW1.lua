-- Only requirement allowed
require("hdf5")

cmd = torch.CmdLine()

-- Cmd Args
cmd:option('-datafile', '', 'data file')
cmd:option('-classifier', 'nb', 'classifier to use')

-- Hyperparameters
-- ...
cmd:option('-alpha', 0, 'smoothing value')
cmd:option('-lambda', 0, 'L2 regularization value')


-- predicts the class from the linear model specified by the parameters W and b
-- note: x is in sparse form
function predict_class(x, W, b)
	local y_hat = b:clone()
	local x = x[x:gt(1)]
	
	for i = 1, x:size(1) do
		y_hat:add(W:index(2, x:long()):sum(2))
	end
	
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
	
	local predictions = predict_test_classes(W, b)
	for i = 1, predictions:size(1) do
		f:write(predictions[i], '\n')
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

function softmax(x, W, b)
	-- local z = 
end

function learn_multiclass_logistic(lambda, m, eta, epochs)
	local W = torch.zeros(nclasses, nfeatures)
	local b = torch.zeros(nclasses)
	
	local i = 0
	while i < epochs do
		local sample = torch.multinomial(torch.ones(train_input:size(1)), m, false)
		
		local grad_W = torch.zeros(nclasses, nfeatures)
		local grad_b = torch.zeros(nclasses)
		
		-- for i = 1:sample:size(1) do
		-- 	local x = train_input[i]
			
		-- 	local y_hat = 
		-- 	local class = train_output[sample[i]]
		-- 	grad_b[]
		-- end
		-- sample
		-- gradient mean = 0
		-- for i = 1 to numsamples
			-- compute loss
			-- compute gradient
			-- update gradient mean
		-- update gradient
	end
	
	
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
   alpha = opt.alpha
   lamdba = opt.lambda
   

   local W = torch.DoubleTensor(nclasses, nfeatures)
   local b = torch.DoubleTensor(nclasses)
   
   print(f:read():all())

   -- Train.
   
   W, b = learn_naive_bayes(alpha)
   
   -- Test.
   eval_linear_model(W, b)
   
   local f_predictions = io.open("predictions.txt", "w")
   print_test_predictions(f_predictions, W, b)
   -- print_test_predictions(nil, W, b)
   f_predictions:close()
end

main()
