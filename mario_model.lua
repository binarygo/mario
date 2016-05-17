require "image"
require "nn"
require "torch"

require "mario_util"
require "util"

local ModelTest = {}

function ModelTest:new()
  local o = {
    _epoch = 0,
    _squeue = nil,
  }
  setmetatable(o, self)
  self.__index = self
  return o
end

function ModelTest:squeue_size()
  return 4
end

function ModelTest:num_sticky_frames()
  return 6
end

function ModelTest:startEpoch(squeue)
  self._epoch = self._epoch + 1
  self._squeue = squeue
end

function ModelTest:selectAction()
  return torch.random(0, 63)
end

function ModelTest:feedback(squeue, mario_dies, level_clear)
  self._squeue = squeue
end

function ModelTest:endEpoch()
  f = torch.DiskFile("squeue.data", "w")
  f:writeObject(self._squeue)
  f:close()
  return false
end

--------------------------------------------------------------------------------

local S_SCREEN_WIDTH = 96
local S_SCREEN_HEIGHT = 96

local function stdGameScreen(screen)
  local x = image.crop(screen, 16, 8, 240, 232)
  return image.rgb2yuv(image.scale(x, S_SCREEN_WIDTH, S_SCREEN_HEIGHT))
end

local ModelLinear = {}

function ModelLinear:new(mode, load_from, save_to)
  local m, x, dx
  if load_from then
    print('Loading model from '..load_from)
    m = torch.load(load_from)
  else
    m = nn.Sequential()
    local input_dim = S_SCREEN_WIDTH * S_SCREEN_HEIGHT * self:squeue_size() + 6
    local output_dim = 1
    m:add(nn.Linear(input_dim, output_dim))
  end

  x, dx = m:getParameters()
  x:zero()
  -- x:copy(torch.randn(x:size()):float())

  local is_train = (mode == "train")
  if is_train then
    m:training()
  else
    m:evaluate()    
  end
    
  local o = {
    _is_train = is_train,
    _load_from = load_from,
    _save_to = save_to,
    _m = m, -- model
    _x = x, -- model parameters
    _dx = dx, -- model grad parameters
    _eps = 0.9, -- eps-greedy
    _gamma = 0.9, -- reward discount
    _alpha = 0.01, -- learning rate
    _epoch = 0,
    _current_state = nil,
  }
  setmetatable(o, self)
  self.__index = self
  return o
end

function ModelLinear:squeue_size()
  return 4
end

function ModelLinear:num_sticky_frames()
  return 3
end

function ModelLinear:startEpoch(squeue)
  self._epoch = self._epoch + 1
  self._current_state = self:_zipState(squeue)
end

function ModelLinear:_zipState(squeue)
  local ss = {}
  for i = 1, self:squeue_size() do
    local s = squeue[i]
    if s then
      local screen = stdGameScreen(s[3])[1]
      ss[#ss + 1] = screen:view(screen:nElement())
    else
      ss[#ss + 1] = ss[#ss]
    end
  end
  return torch.cat(ss)
end

function ModelLinear:_actionTensor(a)
  a = util.bool2IntArray(mario_util.decodeJoypadInput(a))
  return torch.FloatTensor({
      a.up, a.down, a.left, a.right, a.A, a.B
  })
end

function ModelLinear:_evalQ(zipped_state, a)
  local input = torch.cat(zipped_state, self:_actionTensor(a))
  return self._m:forward(input):squeeze(), input
end

function ModelLinear:_maxQ(zipped_state)
  local max_q = nil
  local max_a = nil
  for a = 0, 63 do
    local q, input = self:_evalQ(zipped_state, a)
    if not max_q or q > max_q then
      max_q = q
      max_a = a
    end
  end
  return max_q, max_a
end

function ModelLinear:_randomAction()
  return torch.random(0, 63)
end

function ModelLinear:_greedyAction()
  local max_q, max_a = self:_maxQ(self._current_state)
  return max_a
end

function ModelLinear:selectAction()
  if self._is_train and torch.uniform() > self._eps then
    print("Explore action:")
    return self:_randomAction()
  end
  print("Greedy action:")
  return self:_greedyAction()  
end

function ModelLinear:feedback(squeue, mario_dies, level_clear)
  local s = self._current_state
  local a = squeue[#squeue][1]
  local r = squeue[#squeue][2]
  local sp = self:_zipState(squeue)

  print("Delta reward: "..r)
  
  -- update current state to be the next state
  self._current_state = sp
  
  if self._is_train then
    local target_q = r + self._gamma * self:_maxQ(sp)
    local q, input = self:_evalQ(s, a)
    print("q        = "..q)
    print("target_q = "..target_q)
    self._dx:zero()
    self._m:backward(input, torch.ones(1):float())
    self._x:add(self._dx * self._alpha * (target_q - q))
  end
end

function ModelLinear:endEpoch()
  if self._save_to then
    print('Saving model to '..self._save_to)
    torch.save(self._save_to, self._m)
  end
  if self._is_train then
    return true
  else
    return self._epoch < 1
  end
end

mario_model = {
  ModelTest = ModelTest,
  ModelLinear = ModelLinear,
}
return mario_model