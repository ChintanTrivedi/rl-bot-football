import gfootball.env as football_env
import numpy as np

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K

clipping_val = 0.2
critic_discount = 0.5
entropy_beta = 0.001
gamma = 0.99
lmbda = 0.95


def get_advantages(values, masks, rewards):
    returns = []
    gae = 0
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * values[i + 1] * masks[i] - values[i]
        gae = delta + gamma * lmbda * masks[i] * gae
        returns.insert(0, gae + values[i])

    adv = np.array(returns) - values[:-1]
    return (adv - np.mean(adv)) / (np.std(adv) + 1e-8)


def ppo_loss(oldpolicy_probs, advantages, rewards, values):
    def loss(y_true, y_pred):
        newpolicy_probs = y_true * y_pred
        ratio = K.exp(K.log(newpolicy_probs) - K.log(oldpolicy_probs))
        p1 = ratio * advantages
        p2 = K.clip(ratio, min_value=1 - clipping_val, max_value=1 + clipping_val) * advantages
        actor_loss = -K.mean(K.minimum(p1, p2))
        critic_loss = K.mean(K.square(rewards - values))
        total_loss = critic_discount * critic_loss + actor_loss - entropy_beta * K.mean(-(
                newpolicy_probs * K.log(newpolicy_probs)))
        return total_loss

    return loss


def get_model_actor(input_dims, output_dims):
    state_input = Input(shape=input_dims)
    oldpolicy_probs = Input(shape=(1, output_dims,))
    advantages = Input(shape=(1, 1,))
    rewards = Input(shape=(1, 1,))
    values = Input(shape=(1, 1,))

    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(state_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # Classification block
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    out_actions = Dense(n_actions, activation='softmax', name='predictions')(x)

    model = Model(inputs=[state_input, oldpolicy_probs, advantages, rewards, values],
                  outputs=[out_actions])
    model.compile(optimizer=Adam(lr=1e-4), loss=[ppo_loss(
        oldpolicy_probs=oldpolicy_probs,
        advantages=advantages,
        rewards=rewards,
        values=values)])
    return model


def get_model_critic(input_dims):
    state_input = Input(shape=input_dims)

    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(state_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # Classification block
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    out_actions = Dense(1, activation='tanh')(x)

    model = Model(inputs=[state_input], outputs=[out_actions])
    model.compile(optimizer=Adam(lr=1e-4), loss='mse')
    return model


def test_reward():
    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        state_input = K.expand_dims(state, 0)
        action_probs = model_actor.predict([state_input, dummy_n, dummy_1, dummy_1, dummy_1], steps=1)
        action = np.argmax(action_probs)
        next_state, reward, done, _ = env.step(action)
        state = next_state
        total_reward += reward
    return total_reward


env = football_env.create_environment(
    env_name='academy_empty_goal_close', representation='pixels', render=True)

state = env.reset()
state_dims = env.observation_space.shape
n_actions = env.action_space.n

dummy_n = np.zeros((1, 1, n_actions))
dummy_1 = np.zeros((1, 1, 1))

model_actor = get_model_actor(input_dims=state_dims, output_dims=n_actions)
model_critic = get_model_critic(input_dims=state_dims)

ppo_steps = 128
target_reached = False
best_reward = 0
iters = 0
max_iters = 20

while not target_reached:

    states = []
    actions = []
    values = []
    masks = []
    rewards = []
    action_probs = []
    state_input = None

    for _ in range(ppo_steps):
        state_input = K.expand_dims(state, 0)
        action_dist = model_actor.predict([state_input, dummy_n, dummy_1, dummy_1, dummy_1], steps=1)
        print(np.argmax(action_dist))
        q_value = model_critic.predict([state_input], steps=1)
        action = np.argmax(action_dist)
        # action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        mask = not done
        states.append(state)
        actions.append(action)
        values.append(q_value)
        masks.append(mask)
        rewards.append(reward)
        action_probs.append(action_dist)

        state = observation
        if done:
            env.reset()

    q_value = model_critic.predict(state_input, steps=1)
    values.append(q_value)

    advantages = get_advantages(values, masks, rewards)
    model_actor.fit([states, action_probs, advantages, np.reshape(rewards, newshape=(-1, 1, 1)), values[:-1]],
                    [np.reshape(action_probs, newshape=(-1, n_actions))], verbose=True)
    model_critic.fit([states], [rewards], shuffle=True, epochs=8, verbose=True)

    avg_reward = np.mean([test_reward() for _ in range(10)])
    if avg_reward > best_reward:
        print('best reward=' + str(avg_reward))
        model_actor.save('model_actor_{}_{}.hdf5'.format(iters, avg_reward))
        model_critic.save('model_critic_{}_{}.hdf5'.format(iters, avg_reward))
        best_reward = avg_reward
    if best_reward > 0.9 or iters > max_iters:
        target_reached = True
    iters += 1

env.close()
