1. Dependency
   torch
   cutorch
   cunn
   cudnn
   NVIDIA cuDNN
   fceux SDL version: http://www.fceux.com/web/download.html

2. cd mario
   open fceux UI
   File->Load Lua Script, Load "bot.lua"
   File->Open ROM, open super_mario_bros.nes

3. mario_q_model.lua implements the deep q learning algorithm proposed in
   http://arxiv.org/pdf/1312.5602.pdf

   The entire game is described as
   {s(0), a(0), r(1), s(1), a(1), r(2), s(2), ..., a(n-1), r(n), s(n)} where
   * s(t) = {x(t-3), x(t-2), x(t-1), x(t)}, x(t) is the screenshot at time t,
     ideally, s(t) should be {x(0), x(1), ..., x(t)}
   * x(0) is the start screen and x(n) is the end screen (level clear or
     mario dies or time up).
   * a(t) is the action (up, down, left, right, A, B or their combinations)
     taken at time t.
   * r(t+1) is the score increase after taken action a(t). For example, mario
     grabs a coin after a jump.

   The goal is to learn the function
   Q(s, a) = E(r(t+1) + g * r(t+2) + g^2 * r(t+3) + ... | s(t) = s, a(t) = a)
   where g is a discount factor.

   One way to learn Q(s, a) is through bellman's equation, i.e.
   the optimal Q(s, a) should satisfy
   Q(s, a) = E(r + g * max_a'Q(s', a') | s, a)
   where s, a leads to a score increase r and state s'. For convenience, We call
   (s, a, r, s') a experience.

   For small # of states and actions , one may learn Q(s, a) as a table
   function. However, given the large # of states for this project, it is
   more efficient to consider Q(s, a; theta) where theta is a set of parameters.

   For each (s, a, r, s') sampled from play experience, one compute
   dQ = r + g * max_a'Q(s', a'; theta) - Q(s, a; theta)
   gradQ = grad of Q(s, a; theta) w.r.t theta
   theta <- theta + alpha * dQ * gradQ where alpha is the learning rate.
   
 