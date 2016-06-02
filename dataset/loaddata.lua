--[[
Loads the training data from the dataset
--]]

require 'nn'
require 'image'
require 'paths'

local DataLoader = torch.class 'DataLoader'

function DataLoader:__init(path)
	-- loads the data loader module given the path of the dataset home
	self.rootPath = path
	-- trsize : size of training data
	-- tssize : size of test data
	self.trsize = 50000
	self.tssize = 10000

	--loading dataset

	--loading training data
	self.trainData = {
		data = torch.Tensor(50000, 3072),
		labels = torch.Tensor(50000),
		size = function() return self.trsize end
	}
	local trainData = self.trainData
	for i = 0,4 do
		-- loading training data batch files
		local subset = torch.load(paths.concat(self.rootPath,'cifar-10-batches-t7/data_batch_' ..(i+1).. '.t7'),'ascii')
		trainData.data[{{i*10000+1,(i+1)*10000}}] = subset.data:t()
		trainData.labels[{{i*10000+1,(i+1)*10000}}] = subset.labels
	end
	-- class labels are from 0 to 9 and torch indexes start from 1
	trainData.labels = trainData.labels + 1

	--loading test data
	local subset = torch.load(paths.concat(self.rootPath,'cifar-10-batches-t7/test_batch.t7'),'ascii')
	self.testData = {
		data = subset.data:t():double(),
		labels = subset.labels[1]:double(),
		size = function() return self.tssize end	
	}
	local testData = self.testData
	testData.labels = testData.labels + 1

	-- reshape data
	trainData.data = trainData.data:reshape(self.trsize,3,32,32)
	testData.data = testData.data:reshape(self.tssize,3,32,32)

	self.trainData = trainData
	self.testData = testData
end

function DataLoader:resizeData(trsize, tssize)
	-- resizes data if working on smaller version
	self.trsize = trsize
	self.tssize = tssize

	-- randomly selects trsize data from Training data
	local trainData = {
		data = torch.Tensor(trsize, 3,32,32),
		labels = torch.Tensor(trsize),
		size = function() return self.trsize end
	}

	local trainDataIndices = torch.randperm(50000)
	for i = 1,trsize do
		trainData.data[i] = self.trainData.data[trainDataIndices[i]]
		trainData.labels[i] = self.trainData.labels[trainDataIndices[i]]
	end

	-- randomly selects tssize data from Testing data
	local testData = {
		data = torch.Tensor(tssize, 3,32,32),
		labels = torch.Tensor(tssize),
		size = function() return self.tssize end
	}

	local testDataIndices = torch.randperm(10000)
	for i = 1,tssize do
		testData.data[i] = self.testData.data[testDataIndices[i]]
		testData.labels[i] = self.testData.labels[testDataIndices[i]]
	end

	self.trainData = trainData
	self.testData = testData
end

function DataLoader:trainGenerator(batchSize)
	-- Sends batches of data
	local batchSize = batchSize or 32
	
	local dataIndices = torch.randperm(self.trsize)
	local batches = dataIndices:split(batchSize)
	local i = 1

	local function iterator()
		-- body
		if i < #batches then
			local currentBatch = batches[i]
			local imgList = torch.Tensor(batchSize,3,32,32)
			local clsList = torch.Tensor(batchSize)
			for j = 1,currentBatch:size(1) do
				imgList[j] = self.trainData.data[currentBatch[j]]
				clsList[j] = self.trainData.labels[currentBatch[j]]
			end
			i = i + 1
			return imgList, clsList
		end
	end
	return iterator
end
