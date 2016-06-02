--[[
Main file
--]]

local opt = {
	backend = 'cunn',
	trsize = 1000,
	vsize = 200,
	tssize = 100,
	maxEpochs = 100,
	learningRate = 0.01
}

-- Load model and criterion
require 'models/alexnet.lua'
net = createModel()

criterion = nn.ClassNLLCriterion()

if opt.backend ~= 'nn' then
	require 'cunn'; require 'cudnn'
	cudnn.fastest = true; cudnn.benchmark = true

	net = net:cuda()
	cudnn.convert(net, cudnn) --Convert the net to cudnn
	criterion = criterion:cuda()
end

-- Load dataset
require 'dataset/loaddata.lua'
dl = DataLoader('/data/')
dl:resizeData(opt.trsize, opt.vsize, opt.tssize)

require 'train.lua'
local trainer = Trainer(net, criterion, dl, opt)

for nEpoch=1,opt.maxEpochs do
	local trainLoss = trainer:train()  --Train on training set
	local valLoss = trainer:validate() --Valiate on valiadation set

	print("Epoch "..nEpoch.." complete => "..trainLoss)
end


