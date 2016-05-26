require "image"
require "torch"
require "cutorch"
require "nn"
require "cunn"
require "cudnn"

require "mario_util"

local _ENABLE_CUDA = true
local _TRAIN_FREQ = 1  -- train every # steps
local _EXP_SAMPLING_FREQ = 1  -- sample experience every # steps
-- TODO: increae to 1000000
local _EXP_CACHE_CAPACITY = 2000  -- experience cache capacity
-- TODO: increase to 32
local _MINIBATCH_SIZE = 16  -- minibatch size
local _DISCOUNT_FACTOR = 0.9
local _SQUEUE_SIZE = 4  -- state queue size
local _NUM_STICKY_FRAMES = 6  -- # sticky frames

local QModel = {}

function QModel:_convNetModel()
  local std_screen_width = 84
  local std_screen_height = 84

  local function stdGameScreen(screen)
    local x = image.crop(screen, 49, 60, 208, 219)
    x = image.rgb2yuv(
      image.scale(x, std_screen_width, std_screen_height))
    return x[{{1}, {}, {}}]
  end

  local model = nn.Sequential()
  model:add(nn.SpatialConvolution(self:squeue_size(),16,8,8,4,4,0,0))  -- 84 --> 20
  model:add(nn.ReLU(true))
  model:add(nn.SpatialConvolution(16,32,4,4,2,2,0,0))  -- 20 --> 9
  model:add(nn.ReLU(true))
  model:add(nn.View(32*9*9))
  model:add(nn.Dropout(0.5))
  model:add(nn.Linear(32*9*9, 256))
  model:add(nn.ReLU(true))
  -- valid button combo
  -- left, right, up, down, jump(A), fire(B)
  -- left + A, right + A, left + B, right + B
  model:add(nn.Linear(256, #mario_util.joypad_input_code_from_simple))

  return {
    model = model,
    stdGameScreen = stdGameScreen
  }
end

function QModel:new(mode, load_from, save_to, log_file)
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
    -- x:copy((torch.randn(x:size()) * 0.1):cuda())
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

    _eps = 1.0,  -- eps greedy
    _learning_rate = 0.0001,  -- learning rate

    _epoch = 0,
    _step = 0,
    _current_state = nil,
    _exp_cache = mario_util.LoopQueue:new(_EXP_CACHE_CAPACITY),
    _num_saves = 0,
    _log_file = log_file
  }

  setmetatable(o, self)
  self.__index = self
  return o
end

function QModel:_log(msg)
  mario_util.log(self._log_file, msg)
end

function QModel:squeue_size()
  return _SQUEUE_SIZE
end

function QModel:num_sticky_frames()
  return _NUM_STICKY_FRAMES
end

function QModel:startEpoch(squeue)
  self._epoch = self._epoch + 1
  self._current_state = self:_zipState(squeue)
end

function QModel:_zipState(squeue)
  local ss = {}
  for i = 1, self:squeue_size() do
    local s = squeue[i]  -- state = (a0, r1, s1)
    if s then
      local screen = self._m.stdGameScreen(s[3])
      if _ENABLE_CUDA then
	screen = screen:cuda()
      end
      ss[#ss + 1] = screen
    else
      ss[#ss + 1] = ss[#ss]
    end
  end
  return torch.cat(ss, 1)
end

function QModel:_evalQ(s, a)
  --  s must be returned from _zipState()
  --  a is an integer in [1, 10]
  local output = self._m.model:forward(s)
  return output[a]
end

function QModel:_maxQ(s)
  --  s must be returned from _zipState()
  local output = self._m.model:forward(s)
  local max_q, max_a = torch.max(output, 1)
  return max_q:squeeze(), max_a:squeeze()
end

function QModel:_randomAction()
  return torch.random(1, #mario_util.joypad_input_code_from_simple)
end

function QModel:_greedyAction()
  local max_q, max_a = self:_maxQ(self._current_state)
  return max_a
end

function QModel:selectAction()
  local a = nil
  if torch.uniform() < self._eps then
    print("Explore action:")
    a = self:_randomAction()
  else
    print("Greedy action:")
    a = self:_greedyAction()  
  end
  return mario_util.joypad_input_code_from_simple[a]
end

function QModel:feedback(squeue, mario_dies, level_clear)
  local s = self._current_state
  local a = mario_util.joypad_input_code_to_simple[squeue[#squeue][1]]
  local r = squeue[#squeue][2]
  local sp = nil
  if not (mario_dies or level_clear) then
    sp = self:_zipState(squeue)
  end

  -- update current state to be the next state
  self._step = self._step + 1
  self._current_state = sp
  
  if not self._is_train then
    return
  end

  -- TODO: annealing schedule
  local anneal_steps = 1000000
  if self._step <= anneal_steps then
    self._eps = 1.0 - self._step * (1.0 - 0.1) / anneal_steps
  else
    self._eps = 0.1
  end

  if 0 == self._step % _EXP_SAMPLING_FREQ then
    self._exp_cache:append({s, a, r, sp})
  end

  print("exp_cache_size = "..self._exp_cache:size())
  if not (0 == self._step % _TRAIN_FREQ and
	  self._exp_cache:isFull()) then
    return
  end

  print("Start training...")
  local samples = mario_util.randSample(
    _MINIBATCH_SIZE, self._exp_cache:size())
  for i, sample in ipairs(samples) do
    local exp = self._exp_cache:at(sample)
    local s, a, r, sp = exp[1], exp[2], exp[3], exp[4]
    local target_q = r
    if sp then
      target_q = r + _DISCOUNT_FACTOR * self:_maxQ(sp)
    end
    local q = self:_evalQ(s, a)  -- *
    local dq = target_q - q
    if dq ~= 0 then
      local grad_output = torch.zeros(
	#mario_util.joypad_input_code_from_simple):float()
      if _ENABLE_CUDA then
	grad_output = grad_output:cuda()
      end
      grad_output[a] = 1
      self._dx:zero()
      -- m.model:forward() has been called at *
      self._m.model:backward(s, grad_output)
      -- TODO: more sophisticated learning methods
      self._x:add(self._dx * 1.0e-4 * dq)
    end
  end
end

function QModel:_saveModel()
  if not self._save_to then
    return
  end
  self._num_saves = self._num_saves + 1
  local id = (self._num_saves - 1) % 5 + 1
  local model_save_to = self._save_to..".model."..id

  self:_log(string.format("|x| = %.2f", torch.norm(self._x)))

  self:_log("Saving model to "..model_save_to)
  torch.save(model_save_to, self._m)
end

function QModel:endEpoch()
  if not self._is_train then
    return self._epoch < 1
  end

  self:_saveModel()
  return true
end

mario_q_model = {
  QModel = QModel
}
return mario_q_model
