local net = nn.Sequential()

net:add(nn.SpatialConvolution(1, 6, 5, 5))

net:add(nn.SpatialMaxPooling(2,2,2,2))

net:add(nn.Tanh())

net:add(nn.SpatialConvolution(6, 16, 5, 5))
net:add(nn.SpatialMaxPooling(2,2,2,2))
net:add(nn.Tanh())

net:add(nn.View(16*5*5))

net:add(nn.Linear(16*5*5, 120))
net:add(nn.Tanh())
net:add(nn.Linear(120, 84))
net:add(nn.Tanh())

net:add(nn.Linear(84, 10))
return net
