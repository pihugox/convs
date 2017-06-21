require 'nn'

local nClasses = 1e3

local function inc(input_size, config) -- inception
   local depthCat = nn.Concat(2)

   local conv1 = nn.Sequential()
   conv1:add(nn.SpatialConvolution(input_size, config[1][1], 1, 1)):add(nn.ReLU(true))
   depthCat:add(conv1)

   local conv3 = nn.Sequential()
   conv3:add(nn.SpatialConvolution(input_size, config[2][1], 1, 1)):add(nn.ReLU(true))
   conv3:add(nn.SpatialConvolution(config[2][1], config[2][2], 3, 3, 1, 1, 1, 1)):add(nn.ReLU(true))
   depthCat:add(conv3)

   local conv5 = nn.Sequential()
   conv5:add(nn.SpatialConvolution(input_size, config[3][1], 1, 1)):add(nn.ReLU(true))
   conv5:add(nn.SpatialConvolution(config[3][1], config[3][2], 5, 5, 1, 1, 2, 2)):add(nn.ReLU(true))
   depthCat:add(conv5)

   local pool = nn.Sequential()
   pool:add(nn.SpatialMaxPooling(config[4][1], config[4][1], 1, 1, 1, 1))
   pool:add(nn.SpatialConvolution(input_size, config[4][2], 1, 1)):add(nn.ReLU(true))
   depthCat:add(pool)

   return depthCat
end

local function fac()
   local conv = nn.Sequential()
   conv:add(nn.Contiguous())
   conv:add(nn.View(-1, 1, 224, 224))
   conv:add(nn.SpatialConvolution(1, 8, 7, 7, 2, 2, 3, 3))

   local depthWiseConv = nn.Parallel(2, 2)
   depthWiseConv:add(conv)         -- R
   depthWiseConv:add(conv:clone()) -- G
   depthWiseConv:add(conv:clone()) -- B

   local factorised = nn.Sequential()
   factorised:add(depthWiseConv):add(nn.ReLU(true))
   factorised:add(nn.SpatialConvolution(24, 64, 1, 1)):add(nn.ReLU(true))

   return factorised
end


local main0 = nn.Sequential()
main0:add(fac()) -- 1
--main0:add(nn.SpatialConvolution(3, 64, 7, 7, 2, 2, 3, 3))
main0:add(nn.SpatialMaxPooling(3, 3, 2, 2):ceil())
main0:add(nn.SpatialConvolution(64, 64, 1, 1)):add(nn.ReLU(true)) -- 2
main0:add(nn.SpatialConvolution(64, 192, 3, 3, 1, 1, 1, 1)):add(nn.ReLU(true)) -- 3
main0:add(nn.SpatialMaxPooling(3,3,2,2):ceil())
main0:add(inc(192, {{ 64}, { 96,128}, {16, 32}, {3, 32}})) -- 4,5 / 3(a)
main0:add(inc(256, {{128}, {128,192}, {32, 96}, {3, 64}})) -- 6,7 / 3(b)
main0:add(nn.SpatialMaxPooling(3, 3, 2, 2):ceil())
main0:add(inc(480, {{192}, { 96,208}, {16, 48}, {3, 64}})) -- 8,9 / 4(a)

local main1 = nn.Sequential()
main1:add(inc(512, {{160}, {112,224}, {24, 64}, {3, 64}})) -- 10,11 / 4(b)
main1:add(inc(512, {{128}, {128,256}, {24, 64}, {3, 64}})) -- 12,13 / 4(c)
main1:add(inc(512, {{112}, {144,288}, {32, 64}, {3, 64}})) -- 14,15 / 4(d)

local main2 = nn.Sequential()
main2:add(inc(528, {{256}, {160,320}, {32,128}, {3,128}})) -- 16,17 / 4(e)
main2:add(nn.SpatialMaxPooling(3, 3, 2, 2):ceil())
main2:add(inc(832, {{256}, {160,320}, {32,128}, {3,128}})) -- 18,19 / 5(a)
main2:add(inc(832, {{384}, {192,384}, {48,128}, {3,128}})) -- 20,21 / 5(b)

local sftMx0 = nn.Sequential() -- softMax0
sftMx0:add(nn.SpatialAveragePooling(5, 5, 3, 3))
sftMx0:add(nn.SpatialConvolution(512, 128, 1, 1)):add(nn.ReLU(true))
sftMx0:add(nn.View(128*4*4):setNumInputDims(3))
sftMx0:add(nn.Linear(128*4*4, 1024)):add(nn.ReLU())
sftMx0:add(nn.Dropout(0.7))
sftMx0:add(nn.Linear(1024, nClasses)):add(nn.ReLU())
sftMx0:add(nn.LogSoftMax())

local sftMx1 = nn.Sequential() -- softMax1
sftMx1:add(nn.SpatialAveragePooling(5, 5, 3, 3))
sftMx1:add(nn.SpatialConvolution(528, 128, 1, 1)):add(nn.ReLU(true))
sftMx1:add(nn.View(128*4*4):setNumInputDims(3))
sftMx1:add(nn.Linear(128*4*4, 1024)):add(nn.ReLU())
sftMx1:add(nn.Dropout(0.7))
sftMx1:add(nn.Linear(1024, nClasses)):add(nn.ReLU())
sftMx1:add(nn.LogSoftMax())

local sftMx2 = nn.Sequential() -- softMax2
sftMx2:add(nn.SpatialAveragePooling(7, 7, 1, 1))
sftMx2:add(nn.View(1024):setNumInputDims(3))
sftMx2:add(nn.Dropout(0.4))
sftMx2:add(nn.Linear(1024, nClasses)):add(nn.ReLU()) -- 22
sftMx2:add(nn.LogSoftMax())


local block2 = nn.Sequential()
block2:add(main2)
block2:add(sftMx2)

local split1 = nn.Concat(2)
split1:add(block2)
split1:add(sftMx1)

local block1 = nn.Sequential()
block1:add(main1)
block1:add(split1)

local split0 = nn.Concat(2)
split0:add(block1)
split0:add(sftMx0)

local block0 = nn.Sequential()
block0:add(main0)
block0:add(split0)

local model = block0

return model
