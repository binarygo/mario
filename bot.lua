require "torch"

require "mario_util"
require "mario_game"
require "mario_test_model"
require "mario_q_model"
require "mario_policy_model"

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
      printMessage(
        mario_util.joypadInputToString(mario_util.decodeJoypadInput(a)))
    end
    mario_game.sandbox:next(a, model:num_sticky_frames())
    model:feedback(mario_game.sandbox:squeue(),
                   mario_game.sandbox:marioDies(),
                   mario_game.sandbox:levelClear())
  end
  if debug then
    printMessage("Game ends!")
  end
  return model:endEpoch()
end

local function main()
  local model_class = mario_policy_model.PolicyLinear
  local model = model_class:new(
    "train", nil, "policy_linear.model")
  while doEpoch(model, true) do
  end
end

main()
