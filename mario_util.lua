require "util"

local function _hasBit(x, p)
  return x % (p + p) >= p
end

local function decodeJoypadInput(input_code)
  -- input_code is an 8-bit integer in [0, 255]
  -- |start|select|up|down|left|right|A|B|
  return {
    start = _hasBit(input_code, 0x80),
    select = _hasBit(input_code, 0x40),
    up = _hasBit(input_code, 0x20),
    down = _hasBit(input_code, 0x10),
    left = _hasBit(input_code, 0x08),
    right = _hasBit(input_code, 0x04),
    A = _hasBit(input_code, 0x02),
    B = _hasBit(input_code, 0x01),
  }
end

local function joypadInputToString(input)
  input = util.bool2IntArray(input)
  ss = {""}
  for i, key in ipairs{
    "start", "select", "up", "down", "left", "right", "A", "B"} do
    ss[#ss + 1] = string.format("%s=%d", key, input[key])
  end
  ss[#ss + 1] = ""
  return table.concat(ss, "|")
end

mario_util = {
  decodeJoypadInput = decodeJoypadInput,
  joypadInputToString = joypadInputToString,
}
return mario_util
