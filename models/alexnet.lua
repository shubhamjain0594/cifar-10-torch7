--[[
Implementation of Alexnet presented by Google 
In the One Weird Trick paper. http://arxiv.org/abs/1404.5997
--]]

require 'nn'

local SpatialConvolution = nn.SpatialConvolution
local SpatialMaxPooling = nn.SpatialMaxPooling

function createModel()
	-- Creates alexnet
	local nbClasses = 10
	local nbChannels = 3

	local features = nn.Sequential()
	features:add(SpatialConvolution(nbChannels,64,3,3,1,1,1,1)) -- 32 -> 32 
	features:add(SpatialMaxPooling(2,2,2,2)) -- 32 -> 16
	features:add(nn.ReLU(true))
	features:add(nn.SpatialBatchNormalization(64))

	local classifier = nn.Sequential()
	classifier:add(nn.View(64*16*16))

	classifier:add(nn.Dropout(0.5))
	classifier:add(nn.Linear(64*16*16,1024))
	classifier:add(nn.ReLU(true))

	classifier:add(nn.Linear(1024, nbClasses))
	classifier:add(nn.LogSoftMax())

	local model = nn.Sequential()

	model:add(features):add(classifier)

	return model
end



