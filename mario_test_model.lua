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
  return 5 -- torch.random(0, 63)
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

-------------------------------------------------------------------------------

local TestUct = {}

function TestUct:new()
  local o = {
    _epoch = 0,
    _ref_actions = {},
    _ref_action_cursor = 1,
    _squeue = nil,
    _ref_squeue = nil,
  }
  setmetatable(o, self)
  self.__index = self
  return o
end

function TestUct:squeue_size()
  return 4
end

function TestUct:num_sticky_frames()
  return 6
end

function TestUct:startEpoch(squeue)
  self._epoch = self._epoch + 1
  self._ref_action_cursor = 1
  self._squeue = squeue
end

function TestUct:selectAction()
  local a
  if self._epoch == 1 then
    a = torch.random(0, 63)
    table.insert(self._ref_actions, a)
  else
    a = self._ref_actions[self._ref_action_cursor]
    self._ref_action_cursor = self._ref_action_cursor + 1
  end
  return a
end

function TestUct:feedback(squeue, mario_dies, level_clear)
  self._squeue = squeue
end

local function checkEq(h, a, b)
  if a ~= b then
    return false
  end
  return true
end

function TestUct:endEpoch()
  if self._epoch == 1 then
    self._ref_squeue = self._squeue
  else
    -- check if squeue is exactly the same
    local ref = self._ref_squeue
    local cur = self._squeue
    print(string.format("ref_size = %d", #ref))
    print(string.format("cur_size = %d", #cur))
    if checkEq("size", #ref, #cur) then
      for i = 1,#ref do
        checkEq("action", ref[i][0], cur[i][0])
        checkEq("delta_reward", ref[i][1], cur[i][1])
        checkEq("state", ref[i][2], cur[i][2])
        print(string.format("ref_state = %s", ref[i][2]))
        print(string.format("cur_state = %s", cur[i][2]))
      end
    end
  end
  return self._epoch < 10
end

-------------------------------------------------------------------------------

mario_test_model = {
  TestRand = TestRand,
  TestUct = TestUct,
}
return mario_test_model
