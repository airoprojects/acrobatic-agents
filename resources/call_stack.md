
# Follow this white rabbit

1. main.py -->  update_world() : override
2. rl_util.py -->  update_world() : override
3. rl_world.py --> update() : override
4. rl_world.py --> _update_agents() : override
5. rl_agent.py --> update() : override 
6. rl_agent.py --> _update_new_action() : line ~ 350

```code
# MAJOR CHANGE:
# override action selection by activating the cloning agent:
if override:
  a = override.act(s) # override old action decision with new policy
else:
  a, logp = self._decide_action(s=s, g=g)
```


# For action

1. rl_agent -> _apply_action ->
2. pybullet_deep_mimic_env -> set_action  

