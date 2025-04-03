Part 1
Train the robot to drive to the green goal marker. Use any of the RL algorithms you learnt in the tutorials.

Once loaded you can even continue your training again using the learnt weights, that way you don't have to start from scratch again if you decide you want to train for longer. But keep in mind the epsilon greedy function will start from completely random again so you might want to set epsilon starting value appropriately.

Part 2
Incorporate prior knowledge into the epsilon-greedy function by choosing a non-uniform distribution to sample from when performing exploration. For example, for training flappy bird we used the following to sample flapping actions less often to avoid flying off into the sky during early stages of training:
return np.random.choice(np.array(range(2)), p=[0.9,0.1])

Part 3
Modify the reward to give a bonus of 50 if the goal is reached. You can do this either in the simulate function or directly by modifying the step function in the gym environment code.

Part 4
Add obstacles to the environment. You can do this by modifying the reset function in the gym environment code. For example you can add objects as follows:
self.obstacle = self._p.loadURDF(fileName=<path to urdf file here>,
                   basePosition=[0, 0, 0])

An example urdf file: https://github.com/fredsukkar/simple-car-env-template/blob/main/simple_driving/resources/simplegoal.urdf

Note: you will need to add features to your state so that the agent learns to avoid obstacles. For example, you could add the x, y distance from the agent to the closest obstacle in the environment. Then your state would become: [x_goal, y_goal, x_obstacle, y_obstacle]. 

######################### renders image from third person perspective for validating policy ##############################
env = gym.make("SimpleDriving-v0", apply_api_compatibility=True, renders=False, isDiscrete=True, render_mode='tp_camera')
##########################################################################################################################

######################### renders image from onboard camera ###############################################################
# env = gym.make("SimpleDriving-v0", apply_api_compatibility=True, renders=False, isDiscrete=True, render_mode='fp_camera')
##########################################################################################################################

######################### if running locally you can just render the environment in pybullet's GUI #######################
# env = gym.make("SimpleDriving-v0", apply_api_compatibility=True, renders=True, isDiscrete=True)
##########################################################################################################################

state, info = env.reset()
frames = []
frames.append(env.render())

for i in range(200):
    action = env.action_space.sample()
    state, reward, done, _, info = env.step(action)
    frames.append(env.render())  # if running locally not necessary unless you want to grab onboard camera image
    if done:
        break

env.close()
display_video(frames, framerate=5)  # remove if runnning locally

