require "torch"
util = require "util"

local function decodeJoypadInput(input_code)
  -- input_code is an 8-bit integer in [0, 255]
  -- start|select|A|B|up|down|left|right
  return {
    start = (AND(input_code, 0x80) ~= 0),
    select = (AND(input_code, 0x40) ~= 0),
    A = (AND(input_code, 0x20) ~= 0),
    B = (AND(input_code, 0x10) ~= 0),
    up = (AND(input_code, 0x08) ~= 0),
    down = (AND(input_code, 0x04) ~= 0),
    left = (AND(input_code, 0x02) ~= 0),
    right = (AND(input_code, 0x01) ~= 0)
  }
end

local function joypadInputToString(input)
  ss = {""}
  for i, key in ipairs{
    "start", "select", "A", "B", "up", "down", "left", "right"} do
    ss[#ss + 1] = string.format("%s=%d", key, input[key] and 1 or 0)
  end
  ss[#ss + 1] = ""
  return table.concat(ss, "|")
end

local function setJoypad(player, input_code)
  local input = decodeJoypadInput(input_code)
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
  return torch.ByteTensor(torch.ByteStorage():string(gui.gdscreenshot()))
end

local mario_sandbox = {
  _started = false,
  _mario_lives = nil,
  _world = nil,
  _level = nil,
  _mario_dies = false,
  _level_clear = false,
  _q = nil
}

function mario_sandbox:_reset()
  self._mario_lives = nil
  self._world = nil
  self._level = nil
  self._mario_dies = false
  self._level_clear = false
  self._q = nil
end

function mario_sandbox:_saveGame()
  local save = savestate.object()
  savestate.save(save)
  self._save = save
end

function mario_sandbox:_loadGame()
  if self._save then
    savestate.load(self._save)
  end
end

function mario_sandbox:_updateGameState()
  local mario_lives = getMarioLives()
  local world = getWorld()
  local level = getLevel()

  if self._mario_lives then
    self._mario_dies = mario_lives < self._mario_lives
  end
  if self._world and self._level then
    self._level_clear = (world ~= self._world or level ~= self._level)
  end

  self._mario_lives = mario_lives
  self._world = world
  self._level = level
end

function mario_sandbox:startGame(q_size)
  self:_reset()
  if self._started then
    self:_loadGame()
  else
    emu.speedmode("normal")
    for i = 1, 100 do
      emu.frameadvance()
    end
    setJoypad(1, 0x80)
    skipLiveScreen()
    self:_saveGame()
    self._started = true
  end
  self:_updateGameState()
  self._q = util.LoopQueue:new(q_size)
  -- [s0, (a0, r1, s1), (a1, r2, s2), ...]
  self._q:append({nil, nil, gameScreen()})
end

function mario_sandbox:isGameEnd()
  return self._mario_dies or self._level_clear
end

function mario_sandbox:next(action, num_frames)
  local old_score = getMarioScore()
  for i = 1, num_frames do
    if action then
      setJoypad(1, action)
    end
    emu.frameadvance()
    self:_updateGameState()
    if self:isGameEnd() then
      break
    end
  end
  local s = {nil, getMarioScore() - old_score, gameScreen()}
  self._q:append(s)
  return s
end

local function main()
  while true do
    mario_sandbox:startGame(100)
    emu.message("Game starts!")
    while not mario_sandbox:isGameEnd() do
      -- Every state per 0.1s (for 60Hz rate)
      local a = math.random(0, 63)
      mario_sandbox:next(a, 6)
      emu.message(joypadInputToString(decodeJoypadInput(a)))
    end
    emu.message("Game ends!")
    -- debug
    f = torch.DiskFile("q.data", "w")
    f:writeObject(mario_sandbox._q:array())
    f:close()
    break
  end
end

main()
