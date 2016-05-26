require "image"
require "torch"
require "cutorch"
require "nn"
require "cunn"
require "cudnn"

require "mario_util"

local _ENABLE_CUDA = true
local _LEARNING_RATE = 0.0001  -- learning rate
local _FRAME_SAMPLING_RATE = 0.01  -- sampling rate for elligible frames per epoch
local _EXP_CACHE_CAPACITY = 20  -- experience cache capacity
local _SQUEUE_SIZE = 2  -- state queue size
local _NUM_STICKY_FRAMES = 6  -- # sticky frames

local PolicyModel = {}

function PolicyModel:_mlpModel()
  local std_screen_width = 96
  local std_screen_height = 96

  local function stdGameScreen(screen)
    local x = image.crop(screen, 16, 8, 240, 232)
    return image.rgb2yuv(image.scale(x, std_screen_width, std_screen_height))[1]
  end

  local input_dim = std_screen_width * std_screen_height
  local hidden_dim = 2048
  local output_dim = 64

  local sub_m = nn.Sequential()
  sub_m:add(nn.Reshape(input_dim))
  sub_m:add(nn.Linear(input_dim, hidden_dim))
  sub_m:add(nn.Tanh())
  sub_m:add(nn.Linear(hidden_dim, hidden_dim))
  sub_m:add(nn.Tanh())

  local p = nn.ParallelTable()
  p:add(sub_m)
  for i = 1, self:squeue_size() - 1 do
    local sub_m_copy = sub_m:clone(
      "weight", "bias", "gradWeight", "gradBias")
    p:add(sub_m_copy)
  end

  local model = nn.Sequential()
  model:add(p)
  model:add(nn.JoinTable(1))
  model:add(nn.Dropout(0.5))
  model:add(nn.Linear(hidden_dim * self:squeue_size(), output_dim))
  model:add(nn.LogSoftMax())

  return {
    model = model,
    stdGameScreen = stdGameScreen
  }
end

function PolicyModel:_convNetModel()
  local std_screen_width = 64
  local std_screen_height = 64

  local function stdGameScreen(screen)
    local x = image.crop(screen, 49, 60, 208, 219)
    return image.rgb2yuv(
      image.scale(x, std_screen_width, std_screen_height))[{{1}, {}, {}}]
  end

  local features = nn.Sequential()
  features:add(nn.SpatialConvolution(1,32,5,5,1,1,2,2))  -- 64 --> 64
  features:add(nn.ReLU(true))
  features:add(nn.SpatialMaxPooling(2,2,2,2))            -- 64 --> 32
  features:add(nn.SpatialConvolution(32,32,5,5,1,1,2,2)) -- 32 --> 32
  features:add(nn.ReLU(true))
  features:add(nn.SpatialMaxPooling(2,2,2,2))            -- 32 --> 16

  local classifier = nn.Sequential()
  classifier:add(nn.View(32*16*16))
  classifier:add(nn.Dropout(0.5))

  local sub_m = nn.Sequential()
  sub_m:add(features)
  sub_m:add(classifier)

  local p = nn.ParallelTable()
  p:add(sub_m)
  for i = 1, self:squeue_size() - 1 do
    local sub_m_copy = sub_m:clone(
      "weight", "bias", "gradWeight", "gradBias")
    p:add(sub_m_copy)
  end

  local model = nn.Sequential()
  model:add(p)
  model:add(nn.JoinTable(1))
  model:add(nn.Linear(32*16*16*self:squeue_size(), 2048))
  model:add(nn.Threshold(0, 1.0e-6))
  model:add(nn.Dropout(0.5))
  model:add(nn.Linear(2048, 2048))
  model:add(nn.Threshold(0, 1.0e-6))
  model:add(nn.Linear(2048, 64))
  model:add(nn.LogSoftMax())

  return {
    model = model,
    stdGameScreen = stdGameScreen
  }
end

function PolicyModel:new(mode, load_from, save_to, log_file)
  mario_util.log(log_file, "Start")

  local m, x, dx
  if load_from then
    mario_util.log(log_file, "Loading model from "..load_from)
    m = torch.load(load_from)
  else
    m = self:_convNetModel()
  end

  if _ENABLE_CUDA then
    m.model = m.model:cuda()
  end

  x, dx = m.model:getParameters()
  if not load_from then
    x:zero()
  end

  local is_train = (mode == "train")
  if is_train then
    m.model:training()
  else
    m.model:evaluate()    
  end

  local o = {
    _is_train = is_train,
    _load_from = load_from,
    _save_to = save_to,
    _m = m, -- model
    _x = x, -- model parameters
    _dx = dx, -- model grad parameters

    _epoch = 0,
    _step = 0,
    _current_state = nil,
    _sum_r = 0,
    _exp = {},
    _exp_cache = {},
    _num_saves = 0,
    _log_file = log_file
  }
  setmetatable(o, self)
  self.__index = self
  return o
end

function PolicyModel:_log(msg)
  mario_util.log(self._log_file, msg)
end

function PolicyModel:squeue_size()
  return _SQUEUE_SIZE
end

function PolicyModel:num_sticky_frames()
  return _NUM_STICKY_FRAMES
end
function PolicyModel:startEpoch(squeue)
  self._epoch = self._epoch + 1
  self._step = 0
  self._current_state = self:_zipState(squeue)
  self._sum_r = 0
  self._exp = {}
end

function PolicyModel:_zipState(squeue)
  local ss = {}
  for i = 1, self:squeue_size() do
    local state = squeue[i]  -- state = (a0, r1, s1)
    if state then
      local screen = self._m.stdGameScreen(state[3])
      if _ENABLE_CUDA then
	screen = screen:cuda()
      end
      ss[#ss + 1] = screen
    else
      ss[#ss + 1] = ss[#ss]
    end
  end
  return ss
end

function PolicyModel:selectAction()
  local prob = torch.exp(self._m.model:forward(self._current_state))
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

function PolicyModel:feedback(squeue, mario_dies, level_clear)
  local s = self._current_state
  local a = squeue[#squeue][1]
  local r = squeue[#squeue][2]
  local sp = self:_zipState(squeue)

  -- update current state to be the next state
  self._step = self._step + 1
  self._sum_r = self._sum_r + r
  self._current_state = sp

  print("Step "..self._step..": reward = "..self._sum_r)
  
  if self._is_train and _FRAME_SAMPLING_RATE > torch.uniform() then
    table.insert(self._exp, {self._step, s, a, self._sum_r})
  end
end

function PolicyModel:_saveModel(data)
  if self._save_to then
    self._num_saves = self._num_saves + 1
    local id = (self._num_saves - 1) % 5 + 1
    local data_save_to = self._save_to..".data."..id
    local model_save_to = self._save_to..".model."..id

    self:_log(string.format("|x| = %.2f", torch.norm(self._x)))

    self:_log("Saving data to "..data_save_to)
    torch.save(data_save_to, self._exp_cache)

    self:_log("Saving model to "..model_save_to)
    torch.save(model_save_to, self._m)
  end
end

function PolicyModel:endEpoch()
  if not self._is_train then
    return self._epoch < 1
  end

  if #self._exp > 0 then
    table.insert(self._exp_cache, {
      exp = self._exp,
      end_step = self._step,
      end_r = self._sum_r
    })
  end
  if #self._exp_cache < _EXP_CACHE_CAPACITY then
    return true
  end

  print("Training: #exp_cache = "..#self._exp_cache)

  local total_steps = 0
  local total_rewards = 0.0
  for i, exp in ipairs(self._exp_cache) do
    total_steps = total_steps + exp.end_step
    total_rewards = total_rewards + exp.end_r
  end
  local rho = total_steps > 0 and (total_rewards * 1.0 / total_steps) or 0.0
  self:_log(string.format("total_rewards = %d, total_steps = %d, rho = %f",
			  total_rewards, total_steps, rho))

  local all_exp = {}
  for i, exp in ipairs(self._exp_cache) do
    for j, x in ipairs(exp.exp) do
      local step, s, a, sum_r = x[1], x[2], x[3], x[4]
      local q = exp.end_r - sum_r - (exp.end_step - step) * rho
      table.insert(all_exp, {s, a, q})
    end
  end
  print("#all_exp = "..#all_exp)

  local shuffle = torch.IntTensor()
  torch.randperm(shuffle, #all_exp)
  for i = 1, #all_exp do
    local x = all_exp[shuffle[i]]
    local s, a, q = x[1], x[2], x[3]
    self._dx:zero()
    self._m.model:forward(s)
    local grad_output = torch.zeros(64):float()
    if _ENABLE_CUDA then
      grad_output = grad_output:cuda()
    end
    grad_output[a + 1] = 1
    self._m.model:backward(s, grad_output)

    local dlog_pi = self._dx
    if math.abs(q) ~= 0 then
      self._x:add(dlog_pi * q * _LEARNING_RATE)
      print(string.format("|dlog_pi| = %.2f, E(Q) = %.2f, |x| = %.2f",
			  torch.norm(dlog_pi), q, torch.norm(self._x)))
    end
  end
  self:_saveModel()
  self._exp_cache = {}
  return true
end

mario_policy_model = {
  PolicyModel = PolicyModel
}
return mario_policy_model
