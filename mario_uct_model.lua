require "torch"
require "math"

require "mario_util"

local _NUM_STICKY_FRAMES = 12  -- sticky frames
local _MIN_VISITS_TO_EXPAND_NODE = 5  -- min # visits to expand a node
local _MAX_SIMULATIONS_PER_ROOT = 1000
local _UCT_CONST = 1000.0

local UctModel = {}

function UctModel:new(save_to, log_file)
  mario_util.log(log_file, "Start")
  
  local o = {
    _save_to = save_to,
    _num_saves = 0,
    _log_file = log_file,

    _nodes = {},
    _ans_actions = {},
    
    _epoch = 0,
    _total_reward = 0.0,
    _created_new_arc = false,
    _root_state = nil,
    _root_node = nil,
    _current_node = nil,
    _history = {},
    _ans_action_cursor = 1,
  }
  
  setmetatable(o, self)
  self.__index = self
  return o
end

function UctModel:_log(msg)
  mario_util.log(self._log_file, msg)
end

function UctModel:squeue_size()
  return 1
end

function UctModel:num_sticky_frames()
  return _NUM_STICKY_FRAMES
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

function UctModel:_expand(squeue_elem)
  -- a, r, s
  return squeue_elem[1], squeue_elem[2], squeue_elem[3]
end

-- Returns true if not inited or first time inited.
function UctModel:_delayedInit(squeue)
  if self._ans_action_cursor then
    if self._ans_action_cursor > #self._ans_actions then
      local a, r, s = self:_expand(squeue[1])
      self._root_state = s
      self._root_node = self:_getNode(s)
      self._current_node = self._root_node
      self._history = {{nil, self._current_node}}
      self._ans_action_cursor = nil
    end
    return true
  end
  return false
end

function UctModel:startEpoch(squeue)
  self._epoch = self._epoch + 1
  self._total_reward = 0.0
  self._created_new_arc = false
  self._root_state = nil
  self._root_node = nil
  self._current_node = nil
  self._history = {}
  self._ans_action_cursor = 1

  self:_delayedInit(squeue)
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

function UctModel:_getAnsAction()
  local action = nil
  if self._ans_action_cursor and
     self._ans_action_cursor <= #self._ans_actions then
    action = self._ans_actions[self._ans_action_cursor]
    self._ans_action_cursor = self._ans_action_cursor + 1    
  end
  return action
end

function UctModel:selectAction()
  local action = self:_getAnsAction()
  if action then
    return action
  end
  
  local node = self._current_node
  local legal_actions = mario_util.joypad_input_code_from_simple
  if node and node.narcs == #legal_actions then
    -- if the node has full arcs.
    action = self:_findMaxQAction(node)
  else
    local rest_actions = self:_getRestActions(legal_actions, node)
    action = rest_actions[torch.random(1, #rest_actions)]
  end
  assert(action ~= nil, "selected action must not be nil")
  return action
end

function UctModel:feedback(squeue, mario_dies, level_clear)
  local a, r, s = self:_expand(squeue[1])  -- transition from node to s
  self._total_reward = self._total_reward + r
  if self:_delayedInit(squeue) then
    return
  end
  
  local node = self._current_node
  if not node then
    return
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
    table.insert(self._history, {arc, next_node})
    self._current_node = next_node
  else
    self._current_node = nil
  end
end

function UctModel:_debugNodes()
  print("================= debug ================")
  print("ans actions: ")
  for i, a in ipairs(self._ans_actions) do
    print(string.format("%d, ", a))
  end
  print(string.format("total_reward = %d", self._total_reward))
  local node_count = 1
  for s, node in pairs(self._nodes) do
    print(string.format("node #%d", node_count))
    print(string.format("  nsim = %d", node.nsim))
    for a, arc in pairs(node.arcs) do
      print(string.format("  arc a = %d", a))
      print(string.format("    nsim = %d", arc.nsim))
      print(string.format("    q    = %.2f", arc.q))
      if node.nsim > 0 and arc.nsim > 0 then
        print(string.format("    qx   = %.2f", self:_evaluateQx(node, arc)))
      end
    end
    node_count = node_count + 1
  end
  self:_log(string.format("#nodes = %d", node_count - 1))
end

function UctModel:_saveModel()
  if not self._save_to then
    return
  end
  self._num_saves = self._num_saves + 1
  local id = (self._num_saves - 1) % 5 + 1
  local model_save_to = self._save_to..".model."..id

  self:_log("Saving model to "..model_save_to)
  torch.save(model_save_to, {self._nodes, self._ans_actions})
end

function UctModel:_pruneNodes(new_root_state)
  local visited_states = {}
  local dfs = nil
  dfs = function(s)
    visited_states[s] = true
    local arcs = self._nodes[s].arcs
    for a, arc in pairs(arcs) do
      local next_s = arc.next_state
      if next_s and not visited_states[next_s] then
        dfs(next_s)
      end
    end
  end

  dfs(new_root_state)
  
  local rm_states = {}
  for s, node in pairs(self._nodes) do
    if not visited_states[s] then
      rm_states[s] = true
    end
  end

  for s, v in pairs(rm_states) do
    self._nodes[s] = nil
  end
end

function UctModel:endEpoch()
  for i, h in ipairs(self._history) do
    local arc, node = h[1], h[2]
    if arc then
      arc.nsim = arc.nsim + 1
      arc.q = arc.q + (self._total_reward - arc.q) * 1.0 / arc.nsim
    end
    if node then
      node.nsim = node.nsim + 1
    end
  end

  if self._root_node.nsim >= _MAX_SIMULATIONS_PER_ROOT then
    local ans_action = self:_findMaxNSimAction(self._root_node)
    if ans_action then
      table.insert(self._ans_actions, ans_action)
      self:_pruneNodes(self._root_node.arcs[ans_action].next_state)
    end
  end
  
  self:_log(string.format(
              "epoch = %d, reward = %.2f, #ans_actions = %d",
              self._epoch, self._total_reward, #self._ans_actions))
  self:_debugNodes()
  self:_saveModel()
  return true
end

mario_uct_model = {
  UctModel = UctModel
}
return mario_uct_model
