# RL agent for the Google Football environment

## Setup instructions
Tested on Ubuntu 18.04 and a single NVIDIA GPU.
1. Get the Google Research Football environment up and running using [these instructions](https://github.com/google-research/football#installation). This repository uses the gpu version of tensorflow/gfootball.
2. Install Keras using `pip3 install Keras`.

## Training 
1. Execute `python3 train.py` script to start the PPO training loop.

## Render on remote display server
To render the game screen on a remote display (eg. if using Google Colab), execute the instructions in `display_server.sh`. For more information, check out [this](https://github.com/google-research/football/issues/34) thread.

### Acknowledgements
1. [PPO-Keras](https://github.com/LuEE-C/PPO-Keras)
2. [PPO-PyTorch](https://github.com/colinskow/move37/tree/master/ppo)
3. [Roboschool environment PPO tutorial](https://www.youtube.com/watch?v=WxQfQW48A4A)
