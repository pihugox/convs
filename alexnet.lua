local fb1 = nn.Sequential() -- branch 1
fb1:add(nn.SpatialConvolution(3,48,11,11,4,4,2,2))       -- 224 -> 55
fb1:add(nn.ReLU(true))
fb1:add(nn.SpatialMaxPooling(3,3,2,2))                   -- 55 ->  27
fb1:add(nn.SpatialConvolution(48,128,5,5,1,1,2,2))       --  27 -> 27
fb1:add(nn.ReLU(true))
fb1:add(nn.SpatialMaxPooling(3,3,2,2))                   --  27 ->  13
fb1:add(nn.SpatialConvolution(128,192,3,3,1,1,1,1))      --  13 ->  13
fb1:add(nn.ReLU(true))
fb1:add(nn.SpatialConvolution(192,192,3,3,1,1,1,1))      --  13 ->  13
fb1:add(nn.ReLU(true))
fb1:add(nn.SpatialConvolution(192,128,3,3,1,1,1,1))      --  13 ->  13
fb1:add(nn.ReLU(true))
fb1:add(nn.SpatialMaxPooling(3,3,2,2))                   -- 13 -> 6

local fb2 = fb1:clone() -- branch 2
for k,v in ipairs(fb2:findModules('nn.SpatialConvolution')) do
   v:reset() 
end

local features = nn.Concat(2)
features:add(fb1)
features:add(fb2)

local nClasses = 1e3
local classifier = nn.Sequential()
classifier:add(nn.View(256*6*6))
classifier:add(nn.Dropout(0.5))
classifier:add(nn.Linear(256*6*6, 4096))
classifier:add(nn.Threshold(0, 1e-6))
classifier:add(nn.Dropout(0.5))
classifier:add(nn.Linear(4096, 4096))
classifier:add(nn.Threshold(0, 1e-6))
classifier:add(nn.Linear(4096, nClasses))
classifier:add(nn.LogSoftMax())

local model = nn.Sequential():add(features):add(classifier)
return model
