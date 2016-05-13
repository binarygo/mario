local LoopQueue = {}

function LoopQueue:new(max_n)
  max_n = math.max(1, max_n)
  local o = {
    _max_n = max_n,
    _full = false,
    _next = 1
  }
  setmetatable(o, self)
  self.__index = self
  return o
end

function LoopQueue:append(x)
  if not self._full and self._next == self._max_n then
    self._full = true
  end
  self[self._next] = x
  self._next = self._next % self._max_n + 1
end

function LoopQueue:size()
  return self._full and self._max_n or self._next - 1
end

function LoopQueue:at(i)
  if self._full and i <= self._max_n then
    return self[(self._next + i - 2) % self._max_n + 1]
  elseif not self._full and i <= self._next - 1 then
    return self[i]
  end
end

function LoopQueue:array()
  local result = {}
  for i, v in self:xarray() do
    result[#result + 1] = v
  end
  return result
end

function LoopQueue:xarray()
  local i = 0
  return function ()
    i = i + 1
    v = self:at(i)
    if v then
      return i, v
    end
  end
end

local function bool2IntArray(a)
  local result = {}
  for k, v in pairs(a) do
    result[k] = v and 1 or 0
  end
  return result
end

util = {
  LoopQueue = LoopQueue,
  bool2IntArray = bool2IntArray,
}
return util
