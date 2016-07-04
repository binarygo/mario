require "torch"
require "math"
md5 = require "md5"

require "mario_util"

local SCREEN_WIDTH = 256
local SCREEN_HEIGHT = 240

local STATE_CHOICE = {
  ram = 1,
  ram_md5 = 2,
  screenshot = 3,
}

local sandbox = {
  state_choice = STATE_CHOICE.ram,
  num_skip_frames = 12,
  
  _save = nil,
}

local function setJoypad(player, input_code)
  local input = mario_util.decodeJoypadInput(input_code)
  joypad.set(player, input)
  return input
end

local function getLiveScreenFlag()
  return memory.readbyte(0x0757)
end

local function skipLiveScreen()
  while getLiveScreenFlag() == 0 do
    emu.frameadvance()
  end
  while getLiveScreenFlag() == 1 do
    emu.frameadvance()
  end
end

function sandbox:_saveGame()
  local save = savestate.object()
  savestate.save(save)
  return save
end

function sandbox:_loadGame(save)
  savestate.load(save)
end

function sandbox:_init()
  self._mario_lives = nil
  self._is_game_over = false
end

function sandbox:_update()
  local prev_mario_lives = self._mario_lives
  self._mario_lives = self:getMarioLives()
  if not self._is_game_over then
    self._is_game_over =
      (memory.readbyte(0x000E) == 6) or
      (prev_mario_lives and self._mario_lives < prev_mario_lives)
  end
end

function sandbox:startGame()
  self:_init()
  emu.speedmode("normal")
  if self._save then
    self:_loadGame(self._save)
  else
    for i = 1, 100 do
      emu.frameadvance()
    end
    setJoypad(1, 0x80)
    skipLiveScreen()
    self:setSave()
  end
  self:_update()
end

function sandbox:advance(action)
  for i = 1, self.num_skip_frames do
    if self:isGameOver() then
      return false
    end
    if action then
      setJoypad(1, action)
    end
    emu.frameadvance()
    self:_update()
  end
  return true
end

function sandbox:setTime(t1, t2, t3)
  -- set time to 999
  memory.writebyte(0x07F8, t1)
  memory.writebyte(0x07F9, t2)
  memory.writebyte(0x07FA, t3)  
end

function sandbox:setSave()
  self._save = self:_saveGame()
end

function sandbox:clearSave()
  self._save = nil
end

function sandbox:getWorld()
  return memory.readbyte(0x075F)
end

function sandbox:getLevel()
  return memory.readbyte(0x0760)
end

function sandbox:getMarioScore()
  local score = 0
  local score_addr = 0x07DD
  for i = 1, 6 do
    score = score * 10 + memory.readbyte(score_addr)
    score_addr = score_addr + 1
  end
  return score
end

function sandbox:getMarioLives()
  return memory.readbyte(0x075A)
end

function sandbox:isGameOver()
  return self._is_game_over
end

function sandbox:_screenshot()
  local raw_screen = torch.ByteStorage():string(gui.gdscreenshot())
  local w = SCREEN_WIDTH
  local h = SCREEN_HEIGHT
  return torch.reshape(
    torch.ByteTensor(
      raw_screen, 12,
      torch.LongStorage{h * w, 4})[{{},{2,4}}]:t(), 3, h, w):float() / 255.0
end

function sandbox:_ram(md5_hash)
  local s = memory.readbyterange(0x0000, 0x800)
  return md5_hash and md5.sum(s) or s
end

function sandbox:getState()
  if self.state_choice == STATE_CHOICE.ram_md5 then
    return self:_ram(true)
  elseif self.state_choice == STATE_CHOICE.screenshot then
    return self:_screenshot()
  end
  return self:_ram(false)
end

function sandbox:message(msg)
  emu.message(msg)
end

mario_game = {
  SCREEN_WIDTH = SCREEN_WIDTH,
  SCREEN_HEIGHT = SCREEN_HEIGHT,
  STATE_CHOICE = STATE_CHOICE,
  sandbox = sandbox,
}
return mario_game
