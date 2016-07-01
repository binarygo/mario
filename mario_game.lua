require "torch"
require "math"

require "mario_util"

local SCREEN_WIDTH = 256
local SCREEN_HEIGHT = 240

local function setJoypad(player, input_code)
  local input = mario_util.decodeJoypadInput(input_code)
  joypad.set(player, input)
  return input
end

local function getGameMode()
  return memory.readbyte(0x0770)
end

local function getGameStatus()
  return memory.readbyte(0x0772)
end

local function getLiveScreenFlag()
  return memory.readbyte(0x0757)
end

local function getWorld()
  return memory.readbyte(0x075F)
end

local function getLevel()
  return memory.readbyte(0x0760)
end

local function getMarioLives()
  return memory.readbyte(0x075A)
end

local function getPlayerState()
  return memory.readbyte(0x000E)
end

local function getHScroll()
  return memory.readbyte(0x778)
end

local function getMarioScore()
  local score = 0
  local score_addr = 0x07DD
  for i = 1, 6 do
    score = score * 10 + memory.readbyte(score_addr)
    score_addr = score_addr + 1
  end
  return score
end

local function skipLiveScreen()
  while getLiveScreenFlag() == 0 do
    emu.frameadvance()
  end
  while getLiveScreenFlag() == 1 do
    emu.frameadvance()
  end
end

local function gameScreen()
  local raw_screen = torch.ByteStorage():string(gui.gdscreenshot())
  return torch.reshape(
    torch.ByteTensor(
      raw_screen, 12,
      torch.LongStorage{61440, 4})[{{},{2,4}}]:t(),
    3, SCREEN_HEIGHT, SCREEN_WIDTH):float() / 255.0
end

local function gameRam()
  local s = memory.readbyterange(0x0000, 0x800)
  return s
end

local sandbox = {
  _has_save = false,
  _h = nil,
  _sum_h = 0,
  _mario_lives = nil,
  _world = nil,
  _level = nil,
  _mario_dies = false,
  _level_clear = false,
  _squeue = nil,
  
  max_time = false,
  delayed_start = false,
  use_ram_as_state = false,
}

function sandbox:_reset()
  self._h = nil
  self._sum_h = 0
  self._mario_lives = nil
  self._world = nil
  self._level = nil
  self._mario_dies = false
  self._level_clear = false
  self._squeue = nil
  if self.use_ram_as_state then
    self._get_state_fn = gameRam
  else
    self._get_state_fn = gameScreen
  end
end

function sandbox:_saveGame()
  local save = savestate.object()
  savestate.save(save)
  self._save = save
end

function sandbox:_loadGame()
  if self._save then
    savestate.load(self._save)
  end
end

function sandbox:_updateGameState()
  local h = getHScroll()
  local mario_lives = getMarioLives()
  local world = getWorld()
  local level = getLevel()
  
  if self._h then
    self._sum_h = self._sum_h + math.max(0, h - self._h)
  end
  if self._mario_lives then
    self._mario_dies = mario_lives < self._mario_lives
  end
  if self._world and self._level then
    self._level_clear = (world ~= self._world or level ~= self._level)
  end

  self._h = h
  self._mario_lives = mario_lives
  self._world = world
  self._level = level
end

function sandbox:startGame(squeue_size)
  self:_reset()
  if self._has_save then
    self:_loadGame()
  else
    emu.speedmode("normal")
    for i = 1, 100 do
      emu.frameadvance()
    end
    setJoypad(1, 0x80)
    skipLiveScreen()
    self:_saveGame()
    self._has_save = true
  end
  if self.max_time then
    -- set time to 999
    memory.writebyte(0x07F8, 9)
    memory.writebyte(0x07F9, 9)
    memory.writebyte(0x07FA, 9)
  end
  
  if self.delayed_start then
    for i = 1, 100 do
      emu.frameadvance()
    end
  end

  self:_updateGameState()
  self._squeue = mario_util.LoopQueue:new(squeue_size)
  -- [{nil, nil s0}, (a0, r1, s1), (a1, r2, s2), ...]
  self._squeue:append({nil, nil, self._get_state_fn()})
end

function sandbox:marioDies()
  return self._mario_dies
end

function sandbox:levelClear()
  return self._level_clear
end

function sandbox:isGameEnd()
  return self._mario_dies or self._level_clear
end

function sandbox:getMarioScore()
  return getMarioScore()
end

function sandbox:getSumH()
  return self._sum_h
end

function sandbox:getReward()
  -- return getMarioScore() + self._sum_h
  return getMarioScore()
end

function sandbox:next(action, num_sticky_frames)
  local old_reward = self:getReward()
  for i = 1, num_sticky_frames do
    if action then
      setJoypad(1, action)
    end
    emu.frameadvance()
    self:_updateGameState()
    if self:isGameEnd() then
      break
    end
  end
  local new_reward = self:getReward()
  local s = {
    action,
    math.max(0.0, new_reward - old_reward),
    self._get_state_fn()
  }
  self._squeue:append(s)
  return s
end

function sandbox:squeue()
  return self._squeue:array()
end

function sandbox:message(msg)
  emu.message(msg)
end

mario_game = {
  SCREEN_WIDTH = SCREEN_WIDTH,
  SCREEN_HEIGHT = SCREEN_HEIGHT,
  sandbox = sandbox,
}
return mario_game
