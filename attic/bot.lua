require "torch"

require "mario_util"
require "mario_game"
require "mario_test_model"
require "mario_q_model"
require "mario_policy_model"
require "mario_uct_model"

torch.setdefaulttensortype('torch.FloatTensor')

local function printMessage(msg)
  print(msg)
end

local function doEpoch(model, debug)
  -- The model object must have
  --   model:squeue_size()
  --   model:num_sticky_frames()
  --   model:startEpoch(squeue)
  --   model:selectAction()
  --   model:feedback(squeue)
  --   model:endEpoch()
  mario_game.sandbox:startGame(model:squeue_size())
  model:startEpoch(mario_game.sandbox:squeue())
  if debug then
    printMessage("Game starts!")
  end
  while not mario_game.sandbox:isGameEnd() do
    local a = model:selectAction()
    if debug then
      printMessage(string.format(
        "score = %d, sum_h = %d",
	mario_game.sandbox:getMarioScore(),
	mario_game.sandbox:getSumH()))
      printMessage(
        mario_util.joypadInputToString(mario_util.decodeJoypadInput(a)))
    end
    mario_game.sandbox:next(a, model:num_sticky_frames())
    if not model:feedback(mario_game.sandbox:squeue(),
                          mario_game.sandbox:marioDies(),
                          mario_game.sandbox:levelClear()) then
      break
    end
  end
  if debug then
    printMessage("Game ends!")
  end
  return model:endEpoch()
end

local function q_model_main()
  mario_game.sandbox.delayed_start = true
  local model_class = mario_q_model.QModel
  local model = model_class:new(
    "train", nil, "q_model.sav",
    io.open("q_model.log", "a"))
  while doEpoch(model, true) do
  end
end

local function test_uct_main()
  mario_game.sandbox.use_ram_as_state = true
  local model_class = mario_test_model.TestUct
  local model = model_class:new()
  while doEpoch(model, true) do
  end
end

local function uct_model_main()
  mario_game.sandbox.use_ram_as_state = true
  local model_class = mario_uct_model.UctModel
  local model = model_class:new(
    "uct_model.sav",
    io.open("uct_model.log", "a"))
  while doEpoch(model, true) do
  end  
end

-- q_model_main()
-- test_uct_main()
uct_model_main()
