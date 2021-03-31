import time

import cv2
import numpy as np
from game_control.games.executable_game import ExecutableGame
from game_control.input_controller import KeyboardEvent, KeyboardEvents, KeyboardKey
from game_control.limiter import Limiter


class BrawlStars(ExecutableGame):
    def __init__(self, ldplayer_executable_filepath, fps=2, **kwargs):
        """
        Args:
            fps (int/tuple): Requested number of steps performed per second.
                Will pause in step() to lower number of actions per second.
                Will not pause when fps is too fast for step to keep up.
                Requested fps can be an int or a tuple (indicating a random
                range to choose from, with the top value excluded).
        """
        super().__init__(
            ldplayer_executable_filepath,
            window_name="LDPlayer",
            width=960,
            height=540,
            **kwargs
        )

        self._actions = (
            KeyboardKey.KEY_W,
            KeyboardKey.KEY_A,
            KeyboardKey.KEY_S,
            KeyboardKey.KEY_D,
            KeyboardKey.KEY_E,
            KeyboardKey.KEY_F,
            KeyboardKey.KEY_R,  # Temp nothing
        )

        # top, left, bottom, right
        self._regions = {
            "GAME_SCREEN": (28, 8, 540, 908),
            "BUTTON_EXIT": (476, 497, 522, 566),
            "BUTTON_TRY": (483, 41, 525, 237),
            "BUTTON_PLAY": (345, 539, 384, 688),
            "REWARD_TRY_DAMAGE_PER_SECOND": (67, 848, 89, 900),
        }

        self._limiter = Limiter(fps=fps)

        self.start_app()

    def start_app(self):
        """Starts Brawl Stars app in LDPlayer; returns when it is started.

        Returns:
            Frame: First frame when app has started
        """
        print("starting Brawl Stars")
        return self.grab_frame()

    def stop_app(self):
        """Stops Brawl Stars app; returns when in main screen of LDPlayer.

        Returns:
            Frame: First frame when in main screen of LD Player
        """
        print("stoping Brawl Stars")
        return self.grab_frame()

    def observation_dimensions(self):
        """Calculate the dimensions of region of interest that is analysed.

        Returns:
            tuple: (width, height) of region of interest of actual game.
        """
        (top, left, bottom, right) = self.regions["GAME_SCREEN"]
        width = right - left
        height = bottom - top
        return (width, height)

    @property
    def actions(self):
        return self._actions

    @property
    def regions(self):
        return self._regions

    def observation(self, frame):
        """Cut out the region of interest of the actual game from the full frame.

        Returns:
            np.ndarray: region of interest of the actual game
                TODO: return None or should not be called with frame == None??
        """
        region = self.regions["GAME_SCREEN"]
        if not frame:
            return None
        roi = frame.img[region[0] : region[2], region[1] : region[3]]
        return roi
        # return cv2.resize(roi, self.observation_dimensions())

    def reset(self):
        """Restarts a Brawl Stars event; returns when event is restarted.
            Generic for each event.

        Returns:
            Frame: First frame when event has started
        """
        frame = self.stop_event()
        frame = self.start_event()
        return self.observation(frame)

    def step(self, action):
        """Step function to be wrapped in OpenIA gym environment (GameEnv).
            Take action, obtain new observation, determine reward and if done.
            Generic for each event.

        Args:
            action (KeyboardKey): Action to take

        Returns:
            tuple(np.ndarray, float, bool, dict): with respectively
                * the next observation,
                * the reward of the action that was taken,
                * if the game is done or not
                * some generic info in dict form.
        """
        self._limiter.start()

        # tak action
        self.input_controller.handle_keys([action])

        # Get next observation
        frame = self.grab_frame()
        while frame is None:
            time.sleep(0.5)
            frame = self.grab_frame()
        next_obs = self.observation(frame)

        # Get reward (defined per/in specific event)
        reward = self.reward(frame)

        # Check if done (defined per/in specific event)
        done = self.done(frame)

        (_, step_duration, paused_duration) = self._limiter.stop_and_delay()

        info = {
            "next_observation_timestamp": frame.timestamp,
            "step_duration": step_duration,
            "paused_duration": paused_duration,
        }
        print(info)

        return next_obs, reward, done, info
