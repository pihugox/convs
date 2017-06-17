-- CNN with pooling layers

require 'nn'

local K = 1000
local net = nn.Sequential()

--1
net:add(nn.SpatialConvolution(3, 6, 5, 5, 2, 2, 2, 2))
net:add(nn.ReLU())

--2
net:add(nn.SpatialConvolution(6, 6, 5, 5, 1, 1, 2, 2))
net:add(nn.ReLU())
net:add(nn.SpatialMaxPooling(2, 2, 2, 2))

--3
net:add(nn.SpatialConvolution(6, 6, 5, 5, 1, 1, 2, 2))
net:add(nn.ReLU())
net:add(nn.SpatialMaxPooling(2, 2, 2, 2))


net:add(nn.View(-1))

net:add(nn.Linear(6144, K))

return net
