--[[
Trainer class for training the model
--]]

require 'optim'
require 'xlua'
require 'nn'
require 'image'

local c = require 'trepl.colorize'

local Trainer = torch.class 'Trainer'

function Trainer:__init(model, criterion, dataGen, opt)
	-- Initializes trainer object
	self.model = model
	self.criterion = criterion
	self.opt = opt
	self.optimState = {
		learningRate = opt.learningRate,
		learningRateDecay = 0.0,
		momentum = 0.8,
		nesterov = true,
		dampening = 0.0,
		weightDecay = 0.1
	}
	self.batchSize = 32
	self.nbClasses = 10

	self.params,self.gradParams = model:getParameters()
	self.nEpoch = 1

	self.dataGen = dataGen
	self.confusion = optim.ConfusionMatrix(self.nbClasses)
	if self.opt.backend ~= 'nn' then 
		require 'cudnn'
		require 'cunn'
	end
end


function Trainer:train()
	-- Trains the model for an epoch
	self.optimState.learningRate = self:scheduler(self.nEpoch) 
	print("=> Training epoch #" ..self.nEpoch)
	self.model:training() -- make model trainable as dropouts and batch normalization treated differently in training and validation

	local function feval()
		return self.criterion.output, self.gradParams
	end

	local trainingLoss = 0
	local numBatches = 0
	local tic = torch.tic()
	local count = 0
	self.confusion:zero()

	local input, target, output

	-- Loop over batches
	for input_,target_ in self.dataGen:trainGenerator(self.batchSize) do
		-- Get input and target in batches
		if self.opt.backend ~= 'nn' then
			input = input_:cuda(); target = target_:cuda()
		else
			input = input_; target = target_
		end

		xlua.progress(count+input:size(1), self.dataGen.trsize)
		numBatches = numBatches+1

		-- Backpropogation

		-- Forward pass
		output = self.model:forward(input)
		-- print(output)
		-- print(target)
		local loss = self.criterion:forward(output, target)
		self.confusion:batchAdd(output, target)
		-- Backward pass
		self.model:zeroGradParameters()
		local critGrad = self.criterion:backward(output, target)
		self.model:backward(input, critGrad)	

		-- Update weights
		local _,fs = optim.sgd(feval,self.params, self.optimState)
		trainingLoss = trainingLoss + fs[#fs]
        count = count + input:size(1)
    end

    -- Keeps track of losses and accuracies
    self.confusion:updateValids()
    local trainAcc = self.confusion.totalValid*100

    print(('Train Loss: '..c.cyan'%.4f'..' Accuracy: '..c.cyan'%.2f'..' \t time: %.2f s'):format(trainingLoss/numBatches, trainAcc, torch.toc(tic)))
    -- self:saveImages(input, target, output)

    self.nEpoch = self.nEpoch + 1

    -- returns average loss for this epoch
    return trainingLoss/numBatches
end

function Trainer:scheduler(epoch)
    --[[--
    Learning rate scheduler

    # Parameters
    epoch: int
        adjusts learning rate based on this.
    --]]--
    decay = math.floor((epoch - 1) / 30)
    return self.opt.learningRate* math.pow(0.1, decay)
end