require "torch"

local TestRand = {}

function TestRand:new()
  local o = {
    _epoch = 0,
    _squeue = nil,
  }
  setmetatable(o, self)
  self.__index = self
  return o
end

function TestRand:squeue_size()
  return 4
end

function TestRand:num_sticky_frames()
  return 6
end

function TestRand:startEpoch(squeue)
  self._epoch = self._epoch + 1
  self._squeue = squeue
end

function TestRand:selectAction()
  return torch.random(0, 63)
end

function TestRand:feedback(squeue, mario_dies, level_clear)
  self._squeue = squeue
end

function TestRand:endEpoch()
  f = torch.DiskFile("squeue.data", "w")
  f:writeObject(self._squeue)
  f:close()
  return false
end

mario_test_model = {
  TestRand = TestRand
}
return mario_test_model
