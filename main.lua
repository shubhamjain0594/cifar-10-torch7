--[[
Main file
--]]

local opt = {
	backend = 'nn',
	trsize = 1000,
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
dl = DataLoader('dataset/')
dl:resizeData(opt.trsize,opt.tssize)

require 'train.lua'
local trainer = Trainer(net, criterion, dl, opt)

for nEpoch=1,opt.maxEpochs do
	local trainLoss = trainer:train()  --Train on training set

	print("Epoch "..nEpoch.." complete => "..trainLoss)
end


