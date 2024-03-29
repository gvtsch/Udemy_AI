import collections, gym, numpy as np
from typing import Any, Deque, Tuple

class StartGameWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env) # Konstruktor der Elternklasse
        self.env.reset()

    def reset(self, **kwargs: Any): # kwargs --> Keyword arguments
        self.env.reset()
        observation, _, _, _ = self.env.step(1) # FIRE: Start, sobald man die Leertaste drückt
        return observation

# Frame-Stack, um die Richtung des Balls zu speichern
class FrameStackWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, num_buffer_frames: int):
        super().__init__(env)
        self.num_buffer_frames = num_buffer_frames
        self.frames: Deque = collections.deque(maxlen=self.num_buffer_frames)
        low = np.repeat(
            self.observation_space.low[np.newaxis, ...],
            repeats=self.num_buffer_frames, 
            axis=0
        )
        high = np.repeat(
            self.observation_space.low[np.newaxis, ...],
            repeats=self.num_buffer_frames, 
            axis=0
        )
        self.observation_space = gym.spaces.Box(
            low=low,
            high=high,
            dtype=self.observation_space.dtype
        ) 

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        observation, reward, done, info = self.env.step(action)
        self.frames.append(observation)
        frame_stack = np.asarray(self.frames, dtype=np.float32) # (4, 84, 84)
        frame_stack = np.moveaxis(frame_stack, source=0, destination=-1) # (84, 84, 4)
        frame_stack = np.expand_dims(frame_stack, axis = 0) # (1, 84, 84, 4)
        return frame_stack, reward, done, info

    def reset(self, **kwargs: Any) -> np.ndarray:
        self.env.reset(**kwargs)
        self.frames: Deque = collections.deque(maxlen=self.num_buffer_frames)
        for _ in range(self.num_buffer_frames):
            self.frames.append(np.zeros(shape=(84, 84), dtype=np.float32))
        frame_stack = np.zeros(shape=(1, 84, 84, 4), dtype=np.float32)
        return frame_stack

def make_env(game: str, num_buffer_frames: int):
    env = gym.make(game)
    env = gym.wrappers.AtariPreprocessing(
        env=env,
        noop_max=30, # Max die ersten 30 Frames keine Aktion machen
        frame_skip=4, # Aktion für 4 Frames halten
        screen_size=84,
        terminal_on_life_loss=False,
        grayscale_obs=True,
        scale_obs=True
    )
    env = FrameStackWrapper(env, num_buffer_frames)
    env = StartGameWrapper(env)
    return env

if __name__ == "__main__":
    game = "PongNoFrameskip-v4"
    num_buffer_frames = 4
    env = make_env(game, num_buffer_frames)