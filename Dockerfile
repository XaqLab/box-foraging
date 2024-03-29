FROM zheli21/pytorch:1.13.1-cp310-cuda117-2204 AS base
RUN python -m pip install -U pip stable-baselines3

FROM base as git-repos
RUN mkdir /root/.ssh/
COPY id_ed25519 /root/.ssh/id_ed25519
RUN touch /root/.ssh/known_hosts
RUN ssh-keyscan github.com >> /root/.ssh/known_hosts
RUN git clone git@github.com:lizhe07/jarvis.git
RUN git clone git@github.com:XaqLab/irc-gym.git
RUN git clone git@github.com:XaqLab/box-foraging.git

FROM base as final
COPY --from=git-repos /jarvis /jarvis
RUN pip install /jarvis
COPY --from=git-repos /irc-gym /irc-gym
RUN pip install /irc-gym
COPY --from=git-repos /box-foraging /box-foraging
WORKDIR /box-foraging
