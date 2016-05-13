require "torch"

require "mario_game"
require "mario_model"
require "mario_util"

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
  local model = mario_model.ModelTest:new()
  -- local model = mario_model.ModelLinear:new("train", nil, "model.linear")
  while doEpoch(model, true) do
  end
end

main()
