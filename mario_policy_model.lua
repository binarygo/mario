require "image"
require "nn"
require "torch"

require "mario_util"

local STD_SCREEN_WIDTH = 64
local STD_SCREEN_HEIGHT = 64

local function stdGameScreen(screen)
  local x = image.crop(screen, 16, 8, 240, 232)
  return image.rgb2yuv(image.scale(x, STD_SCREEN_WIDTH, STD_SCREEN_HEIGHT))[1]
end

local PolicyLinear = {}

function PolicyLinear:new(mode, load_from, save_to)
  local m, x, dx
  if load_from then
    print('Loading model from '..load_from)
    m = torch.load(load_from)
  else
    m = nn.Sequential()
    local input_dim = STD_SCREEN_WIDTH * STD_SCREEN_HEIGHT * self:squeue_size()
    local output_dim = 64
    m:add(nn.Linear(input_dim, 2000))
    m:add(nn.Tanh())
    m:add(nn.Linear(2000, output_dim))
    m:add(nn.SoftMax())
  end

  x, dx = m:getParameters()
  x:zero()

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
    _alpha = 0.01, -- learning rate
    _sampling_rate = 0.1, -- sampling rate for elligible frames per epoch
    _epoch = 0,
    _current_state = nil,
    _sum_r = 0,
    _exp_cache = {}
  }
  setmetatable(o, self)
  self.__index = self
  return o
end

function PolicyLinear:squeue_size()
  return 2
end

function PolicyLinear:num_sticky_frames()
  return 3
end

function PolicyLinear:startEpoch(squeue)
  self._epoch = self._epoch + 1
  self._current_state = self:_zipState(squeue)
  self._sum_r = 0
  self._exp_cache = {}
end

function PolicyLinear:_zipState(squeue)
  local ss = {}
  for i = 1, self:squeue_size() do
    local s = squeue[i]
    if s then
      local screen = stdGameScreen(s[3])
      ss[#ss + 1] = screen:view(screen:nElement())
    else
      ss[#ss + 1] = ss[#ss]
    end
  end
  return torch.cat(ss)
end

function PolicyLinear:selectAction()
  local prob = self._m:forward(self._current_state)
  local a = nil
  if self._is_train then
    a = torch.sum(torch.lt(torch.cumsum(prob), torch.uniform())) % 64
  else
    max_prob, a = torch.max(prob, 1)
    a = a:squeeze() - 1
  end
  print(string.format(
          "Select action %d with prob = %.2f%%", a, prob[a + 1] * 100.0))
  return a
end

function PolicyLinear:feedback(squeue, mario_dies, level_clear)
  local s = self._current_state
  local a = squeue[#squeue][1]
  local r = squeue[#squeue][2]
  local sp = self:_zipState(squeue)

  -- update current state to be the next state
  self._current_state = sp
  self._sum_r = self._sum_r + r + (mario_dies and -100 or 0)

  print("Reward: "..self._sum_r)
  
  if self._is_train and self._sampling_rate > torch.uniform() then
    self._dx:zero()
    self._m:forward(s)
    grad_output = torch.zeros(64):float()
    grad_output[a + 1] = 1
    self._m:backward(s, grad_output)
    table.insert(self._exp_cache, {self._dx:clone(), self._sum_r})
  end
end

function PolicyLinear:endEpoch()
  local cont = false
  if self._is_train then
    print("End epoch training: #exp_cache = "..#self._exp_cache)
    for i, exp in ipairs(self._exp_cache) do
      dpi, sum_r = exp[1], exp[2]
      q = self._sum_r - sum_r
      if math.abs(q) ~= 0 then
        self._x:add(dpi * q * self._alpha)
        print(string.format("|dpi| = %.2f, E(Q) = %.2f, |x| = %.2f",
                            torch.norm(dpi), q, torch.norm(self._x)))
      end
    end
    cont = true
  else
    cont = self._epoch < 1
  end
  if self._save_to then
    local save_to = self._save_to.."."..(self._epoch % 10)
    print("Saving model to "..save_to)
    torch.save(save_to, self._m)
  end
  return cont
end

mario_policy_model = {
  PolicyLinear = PolicyLinear
}
return mario_policy_model
