require "torch"
require "math"

require "mario_util"

local _MIN_VISITS_TO_EXPAND_NODE = 1  -- min # visits to expand a node
local _MAX_SIMULATIONS = 100
local _MAX_DEPTH = 80
local _UCT_CONST = 100.0
local _MAX_Q = true

local UctModel = {}

function UctModel:new(save_to, log_file)
  mario_util.log(log_file, "Start")
  
  local o = {
    _save_to = save_to,
    _num_saves = 0,
    _log_file = log_file,

    _nodes = {},
    _ans_actions = {},
    _ans_action_cursor = 1,
    
    _epoch = 0,
  }
  
  setmetatable(o, self)
  self.__index = self
  return o
end

function UctModel:_log(msg)
  mario_util.log(self._log_file, msg)
end

function UctModel:_getNode(state)
  local node = self._nodes[state]
  if not node then
    node = {
      nsim = 0,  -- num simulations
      arcs = {},  -- arcs[action] = {q = ..., nsim = ...}
      narcs = 0,
    }
    self._nodes[state] = node
  end
  return node
end

function UctModel:_getArc(node, action)
  local arc = node.arcs[action]
  if not arc then
    arc = {
      nsim = 0,
      q = 0.0,
      next_state = nil,
    }
    node.arcs[action] = arc
    node.narcs = node.narcs + 1
  end
  return arc
end

function UctModel:_init(s)
  if s then
    self._depth = 1
    self._created_new_arc = false
    self._root_state = s
    self._root_node = self:_getNode(s)
    self._current_node = self._root_node
    self._trace = {{nil, self._root_node}}
  else
    self._depth = nil
    self._created_new_arc = nil
    self._root_state = nil
    self._root_node = nil
    self._current_node = nil
    self._trace = nil
  end
end

function UctModel:startEpoch(s)
  self._epoch = self._epoch + 1
  self._score = nil
  self._is_pre_stage = (self._ans_action_cursor <= #self._ans_actions)

  if self._is_pre_stage then
    self:_init(nil)
    self._tmp_ans_action_cursor = self._ans_action_cursor
  else
    self:_init(s)
  end
end

function UctModel:_evaluateQx(node, arc)
  return arc.q + _UCT_CONST * math.sqrt(math.log(node.nsim) / arc.nsim)
end

function UctModel:_findMaxQAction(node)
  assert(node.nsim > 0, "node must have been visited at least once")
  local max_qx = nil
  local action = nil
  for a, arc in pairs(node.arcs) do
    assert(arc.nsim > 0, "arc must have been visited at least once")
    local qx = self:_evaluateQx(node, arc)
    if not max_qx or max_qx < qx then
      max_qx, action = qx, a
    end
  end
  return action
end

function UctModel:_findMaxNSimAction(node)
  local max_nsim = nil
  local action = nil
  for a, arc in pairs(node.arcs) do
    if not max_nsim or max_nsim < arc.nsim then
      max_nsim, action = arc.nsim, a
    end
  end
  return action
end

function UctModel:_getRestActions(legal_actions, node)
  if node and node.narcs > 0 then
    local rest_actions = {}
    for i, a in ipairs(legal_actions) do
      if not node.arcs[a] then
        table.insert(rest_actions, a)
      end
    end
    return rest_actions
  end
  return legal_actions
end

function UctModel:selectAction()
  local action = nil
  if self._is_pre_stage then
    action = self._ans_actions[self._tmp_ans_action_cursor]
    self._tmp_ans_action_cursor = self._tmp_ans_action_cursor + 1
    print("Ans action")
  else
    local node = self._current_node
    local legal_actions = mario_util.joypad_input_code_from_simple
    if node and node.narcs == #legal_actions then
      -- if the node has full arcs.
      action = self:_findMaxQAction(node)
      print("UCB action")
    else
      local rest_actions = self:_getRestActions(legal_actions, node)
      action = rest_actions[torch.random(1, #rest_actions)]
      print("Random action")
    end
  end
  assert(action ~= nil, "selected action must not be nil")
  return action
end

function UctModel:feedback(a, r, s, is_game_over)
  self._score = is_game_over and 0 or r

  if self._is_pre_stage then
    if self._tmp_ans_action_cursor > self._ans_action_cursor then
      self._ans_action_cursor = self._tmp_ans_action_cursor
      self._tmp_ans_action_cursor = nil
      self:_init(s)
      self._is_pre_stage = false
      return true  -- update game save
    end
  end

  print("depth = "..self._depth)
  self._depth = self._depth + 1
  
  local node = self._current_node
  if not node then  -- random walk, not update trace
    return false  -- not update game save
  end
  
  local arc = node.arcs[a]
  if not self._created_new_arc and not arc and
     (node == self._root_node or node.nsim >= _MIN_VISITS_TO_EXPAND_NODE) then
    -- if the node is expandable
    self:_getNode(s)
    arc = self:_getArc(node, a)
    arc.next_state = s
    self._created_new_arc = true
  end

  if arc then
    local next_node = self:_getNode(s)
    table.insert(self._trace, {arc, next_node})
    self._current_node = next_node
  else
    self._current_node = nil
  end
  return false  -- not update game save
end

function UctModel:stop()
  if not self._is_pre_stage then
    return self._depth > _MAX_DEPTH
  end
  return false
end

function UctModel:_debugNodes()
  self:_log("================= debug ================")
  self:_log("ans actions: ")
  for i, a in ipairs(self._ans_actions) do
    self:_log(string.format("%d, ", a))
  end
  self:_log(string.format("total_reward = %d", self._score))
  local node_count = 1
  for s, node in pairs(self._nodes) do
    self:_log(string.format("node #%d", node_count))
    self:_log(string.format("  nsim = %d", node.nsim))
    for a, arc in pairs(node.arcs) do
      self:_log(string.format("  arc a = %d", a))
      self:_log(string.format("    nsim = %d", arc.nsim))
      self:_log(string.format("    q    = %.2f", arc.q))
      if node.nsim > 0 and arc.nsim > 0 then
        self:_log(string.format("    qx   = %.2f", self:_evaluateQx(node, arc)))
      end
    end
    node_count = node_count + 1
  end
  self:_log(string.format("#nodes = %d", node_count - 1))
  self:_log("========================================")
end

function UctModel:_saveModel()
  if not self._save_to then
    return
  end
  self._num_saves = self._num_saves + 1
  local id = (self._num_saves - 1) % 5 + 1
  local model_save_to = self._save_to..".model."..id

  self:_log("Saving model to "..model_save_to)
  torch.save(model_save_to, self._ans_actions)
end

function UctModel:endEpoch()
  for i, h in ipairs(self._trace) do
    local arc, node = h[1], h[2]
    if arc then
      arc.nsim = arc.nsim + 1
      if _MAX_Q then
        arc.q = math.max(arc.q, self._score)
      else
        arc.q = arc.q + (self._score - arc.q) * 1.0 / arc.nsim
      end
    end
    if node then
      node.nsim = node.nsim + 1
    end
  end

  if self._root_node.nsim >= _MAX_SIMULATIONS then
    local ans_action = self:_findMaxNSimAction(self._root_node)
    if ans_action then
      table.insert(self._ans_actions, ans_action)
      self._nodes = {}  -- clear all nodes
    end
  end
  
  self:_log(string.format(
              "epoch = %d, score = %.2f, #ans_actions = %d",
              self._epoch, self._score, #self._ans_actions))
  self:_debugNodes()
  self:_saveModel()
  return true
end

mario_uct_model = {
  UctModel = UctModel
}
return mario_uct_model
