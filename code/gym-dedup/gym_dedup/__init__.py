from gym.envs.registration import register

register(
    id='dedup-v0',
    entry_point='gym_dedup.envs:DedupEnv'
)
