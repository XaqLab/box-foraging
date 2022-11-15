# box-foraging
Foraging experiment with color cued boxes

## Inverse Rational Control (IRC)
We assume an agent that performs suboptimally in an environment is due to its incorrect assumption
about the environment, while it actually performs optimally according to the assumed environment.
Therefore it is possible to infer an agents internal model of the environment based on its behavior,
more specifically the observations given to the agent and the actions it takes accordingly. We call
the analysis Inverse Rational Control, and developed the `irc` package for this purpose.

We formalize the environment as a Partially Observable Markov Decision Process (POMDP). Denoting the
environment state, agent action, and observation at time $t$ as $s_t$, $a_t$, $o_t$ respectively,
the state transition is described by the conditional distribution $p(s_{t+1}|s_t, a_t)$, and the
sensory observation is described by $p(o_t|s_t)$.

We hypothesize that an agent makes its decision in POMDP not based on the direct observation $o_t$,
but a vector $b_t$ called belief that uniquely defines a distribution over environment state $s_t$.
Once an internal model of the environment is provided, `irc` can get belief $b_{t+1}$ from $b_t$,
$a_t$ and $o_{t+1}$. The belief update is sample-based and therefore stochastic by nature. Treating
the belief sequence as the states of a Markov Decision Process (MDP), `irc` uses off-the-shelf
reinforcement learning (RL) algorithms to get the optimal policy.

`irc` does not require the user to specify the full distribution $p(s_{t+1}|s_t, a_t)$ and $p(o_t|s_t)$
for the internal model, but instead only a simulator that can samples new environment state and
observation for each time step. The user defined environment will be a standard OpenAI Gym
environment, with some additional utility methods implemented.

### Basic Gym Requirements
A valid gym environment class has to be a subclass of `gym.Env`. It has two properties
`self.observation_space` and `self.action_space`, both are valid `gym.spaces`. Currently the `irc`
package only supports discrete observation and action, namely `self.observation_space` is an
instance of `MultiDiscrete` and `self.action_space` is an instance of `Discrete`.

The environment class also has two methods, `self.reset()` and `self.step()`. The method `reset`
takes no arguments, and returns an observation, e.g. `observation = env.reset()`. The method `step`
takes an action (an integer in current version) as input argument, and returns a 4-tuple
(`observation`, `reward`, `done`, `info`), e.g. `observation, reward, done, info = env.step(action)`.

### Additional Requirements
For `irc` to work, the environment class is required to have the following properties and methods.

First the environment must have the property `self.state_space`, which is generally different from
`self.observation_space` in POMDP. Again, currently `irc` only supports discrete state space, i.e.
`MultiDiscrete`.

Next the environment needs to have methods `self.get_env_param()` and `self.set_env_param()`. The
method `get_env_param` takes no arguments, and returns a tuple of floats for environment parameters,
`env_param = env.get_env_param()`. The method `set_env_param` takes a tuple of floats as input
argument, typically returned by `get_env_param`, and sets the environment parameter to the given
values, `env.set_env_param(env_param)`.

The environemt also needs to have methods `self.get_state()` and `self.set_state()`. The method
`get_state` takes no arguments, and returns a tuple of floats for current state, `state = env.get_state()`.
The method `set_state` takes a tuple of floats as input argument, and sets the environment state to
given values, `env.set_state(state)`.

Optionally, the environment can implement a method `self.query_states()` that is useful for results
analysis. `query_states` takes no arguments and returns a list of states, `states = env.query_states()`.
`irc` can return the probabilities of each state of interest according to the beliefs at each time
point, so that the user can visualize belief traces during an episode.

### Examples
```bash
python demo-train.py env_param=[0.2,0.6,10.] seed=1 num_epochs=8
```
