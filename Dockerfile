FROM zheli21/pytorch:1.12.1-cp39-cuda116-2004 AS base
RUN python -m pip install -U pip stable-baselines3

FROM base as git-repos
RUN mkdir /root/.ssh/
COPY id_ed25519 /root/.ssh/id_ed25519
RUN touch /root/.ssh/known_hosts
RUN ssh-keyscan github.com >> /root/.ssh/known_hosts
RUN git clone -b 0.5 git@github.com:lizhe07/jarvis.git
RUN git clone -b 0.2 git@github.com:XaqLab/irc-gym.git
RUN git clone git@github.com:XaqLab/box-foraging.git

FROM base as final
COPY --from=git-repos /jarvis /jarvis
RUN pip install /jarvis
COPY --from=git-repos /irc-gym /irc-gym
RUN pip install /irc-gym
COPY --from=git-repos /box-foraging /box-foraging
WORKDIR /box-foraging
