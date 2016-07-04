require "torch"

require "mario_util"
require "mario_game"
require "mario_uct_model"

local bot = {
  enable_debug = true,
}

function bot:_debugMessage(msg)
  if self.enable_debug then
    print(msg)
  end
end

function bot:doEpoch(model)
  -- The model object must have
  --   model:startEpoch(s)
  --   model:selectAction()
  --   model:feedback(a, r, s, is_game_over)
  --   model:stop()
  --   model:endEpoch()
  mario_game.sandbox:startGame()
  model:startEpoch(mario_game.sandbox:getState())
  self:_debugMessage("Epoch starts!")
  while not mario_game.sandbox:isGameOver() and not model:stop() do
    local a = model:selectAction()
    self:_debugMessage(string.format("score = %d",
                                     mario_game.sandbox:getMarioScore()))
    self:_debugMessage(mario_util.joypadInputToString(
                         mario_util.decodeJoypadInput(a)))
    mario_game.sandbox:advance(a)
    if model:feedback(a, mario_game.sandbox:getMarioScore(),
                      mario_game.sandbox:getState(),
                      mario_game.sandbox:isGameOver()) then
      mario_game.sandbox:setSave()
      self:_debugMessage("Updated save!")
    end
  end
  self:_debugMessage("Epoch ends!")
  return model:endEpoch()
end

local function uct_main()
  mario_game.sandbox.state_choice = mario_game.STATE_CHOICE.ram_md5
  mario_game.sandbox.num_skpi_frames = 12
  bot.enable_debug = true
  local model = mario_uct_model.UctModel:new(
    "uct_model.sav", io.open("uct_model.log", "a"))
  while bot:doEpoch(model) do
  end
end

uct_main()
