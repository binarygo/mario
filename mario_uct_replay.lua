require "torch"

require "mario_util"
require "mario_game"

local function play_main()
  actions = torch.load("uct_model.sav.model.5")
  
  mario_game.sandbox:reset()
  mario_game.sandbox.num_skpi_frames = 12
  while true do
    mario_game.sandbox:startGame()
    for i, a in ipairs(actions) do
      mario_game.sandbox:advance(a)
    end
  end
end

play_main()
