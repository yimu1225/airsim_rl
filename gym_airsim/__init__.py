from gymnasium.envs.registration import register

register(
	id='AirSimEnv-v42',
	entry_point='gym_airsim.envs:AirSimEnv',
)

register(
	id='AirSimEnv-Gradient-v1',
	entry_point='gym_airsim.envs:AirSimEnvGradientReward',
)
