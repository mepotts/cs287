require("nn")

local mlp = nn.Sequential()

local parallel = nn.ParallelTable()
local tablewords = nn.Sequential()
tablewords:add(nn.LookupTable(2*3, 2))
-- tablewords:add(nn.View(2*2))
parallel:add(tablewords)
local capwords = nn.Sequential()
capwords:add(nn.LookupTable(2*1, 1))
parallel:add(capwords)

mlp:add(parallel)
mlp:add(nn.JoinTable(2))

a = torch.Tensor({{1,2}, {2,3}})
b = torch.Tensor({{2,2}, {1,2}})

-- res = mlp:forward({a, b})
-- print(res)

res2 = tablewords:forward(a)
print(res2)

tablewords:add(nn.View(2*2))
res2 = tablewords:forward(a)
print(res2)

asub = torch.Tensor({{1,2}})
res2 = tablewords:forward(asub)
print(res2)
