import time
from pathlib import Path

import cv2
import numpy as np
from game_control.input_controller import KeyboardEvent, KeyboardEvents, KeyboardKey
from game_control.limiter import Limiter
from game_control.sprite import Sprite

from brawl_stars_gym.ldplayer import LDPlayer


class BrawlStars(LDPlayer):
    BRAWL_STARS_DIR = Path("brawl_stars")

    def __init__(self, ldplayer_executable_filepath, fps=2, **kwargs):
        """
        Args:
            ldplayer_executable_filepath (string): Executable of LDPlayer
            fps (int/tuple): Requested number of steps performed per second.
                Will pause in step() to lower number of actions per second.
                Will not pause when fps is too fast for step to keep up.
                Requested fps can be an int or a tuple (indicating a random
                range to choose from, with the top value excluded).
        """
        # Need fixed size window for region definitions
        super().__init__(ldplayer_executable_filepath, width=960, height=540, **kwargs)

        self._sprites.update(
            Sprite.discover_sprites(
                Path(__file__).parent
                / self.DATA_DIR
                / self.BRAWL_STARS_DIR
                / self.SPRITE_DIR
            )
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

        # self._actions = (
        #     (
        #         "MOVEMENT", (
        #             ("MOVE UP", (KeyboardEvent(KeyboardEvents.DOWN, KeyboardKey.KEY_W)),
        #             ("MOVE LEFT", (KeyboardEvent(KeyboardEvents.DOWN, KeyboardKey.KEY_A)),
        #             ("MOVE DOWN", (KeyboardEvent(KeyboardEvents.DOWN, KeyboardKey.KEY_S)),
        #             ("MOVE RIGHT", (KeyboardEvent(KeyboardEvents.DOWN, KeyboardKey.KEY_D))
        #         )
        #     ),
        #     (
        #         "SHOOTING", (
        #             ("SHOOT", (KeyboardEvent(KeyboardEvents.DOWN, KeyboardKey.KEY_E)),
        #             ("SHOOT SUPER", (KeyboardEvent(KeyboardEvents.DOWN, KeyboardKey.KEY_F)),
        #             ("DON'T SHOOT", ())
        #         )
        #     )
        # )

        # self._actions = {
        #     "MOVEMENT": {
        #         "MOVE UP": [
        #             KeyboardEvent(KeyboardEvents.DOWN, KeyboardKey.KEY_W)
        #         ],
        #         "MOVE LEFT": [
        #             KeyboardEvent(KeyboardEvents.DOWN, KeyboardKey.KEY_A)
        #         ],
        #         "MOVE DOWN": [
        #             KeyboardEvent(KeyboardEvents.DOWN, KeyboardKey.KEY_S)
        #         ],
        #         "MOVE RIGHT": [
        #             KeyboardEvent(KeyboardEvents.DOWN, KeyboardKey.KEY_D)
        #         ],
        #         # "MOVE TOP-LEFT": [
        #         #     KeyboardEvent(KeyboardEvents.DOWN, KeyboardKey.KEY_W),
        #         #     KeyboardEvent(KeyboardEvents.DOWN, KeyboardKey.KEY_A)
        #         # ],
        #         # "MOVE TOP-RIGHT": [
        #         #     KeyboardEvent(KeyboardEvents.DOWN, KeyboardKey.KEY_W),
        #         #     KeyboardEvent(KeyboardEvents.DOWN, KeyboardKey.KEY_D)
        #         # ],
        #         # "MOVE DOWN-LEFT": [
        #         #     KeyboardEvent(KeyboardEvents.DOWN, KeyboardKey.KEY_S),
        #         #     KeyboardEvent(KeyboardEvents.DOWN, KeyboardKey.KEY_A)
        #         # ],
        #         # "MOVE DOWN-RIGHT": [
        #         #     KeyboardEvent(KeyboardEvents.DOWN, KeyboardKey.KEY_S),
        #         #     KeyboardEvent(KeyboardEvents.DOWN, KeyboardKey.KEY_D)
        #         # ],
        #         "DON'T MOVE": []
        #     },
        #     "SHOOTING": {
        #         "SHOOT": [
        #             KeyboardEvent(KeyboardEvents.DOWN, KeyboardKey.KEY_E)
        #         ],
        #         "SHOOT SUPER": [
        #             KeyboardEvent(KeyboardEvents.DOWN, KeyboardKey.KEY_F)
        #         ],
        #         "DON'T SHOOT": []
        #     }
        # }

        # top, left, bottom, right
        self._regions.update(
            {
                "BUTTON_BRAWL_STARS": (103, 315, 185, 384),
                "BUTTON_BRAWLERS": (301, 28, 337, 92),
                "BUTTON_BACK": (34, 13, 84, 91),
                "GAME_SCREEN": (28, 8, 540, 908),
            }
        )

        self._limiter = Limiter(fps=fps)

        self.start_app()

    def start_app(self):
        """Starts Brawl Stars app in LDPlayer; returns when it is started.

        Returns:
            Frame: First frame when started (or failed).

        Raises:
            RuntimeError: when app could not be started
        """
        sprite = self.sprites["SPRITE_BUTTON_BRAWL_STARS"]
        region = self.regions["BUTTON_BRAWL_STARS"]
        found, frame, _ = self._wait_for_sprite(
            sprite, region=region, msg="Waiting for LDPlayer ..."
        )
        if not found:
            raise RuntimeError("Could not find sprite", sprite.name, "in time")
        self.input_controller.click_screen_region(region)

        sprite = self.sprites["SPRITE_BUTTON_BRAWLERS"]
        region = self.regions["BUTTON_BRAWLERS"]
        found, frame, _ = self._wait_for_sprite(
            sprite, region=region, msg="Waiting for Brawl Stars ..."
        )
        if not found:
            from game_control.utilities import extract_roi_from_image

            cv2.imwrite("C:\\Temp\\frame.png", frame.img)
            roi = extract_roi_from_image(frame.img, region)
            cv2.imwrite("C:\\Temp\\roi.png", roi)
            cv2.imwrite("C:\\Temp\\sprite.png", sprite.image_data[..., 0])
            raise RuntimeError("Could not find sprite", sprite.name, "in time")

        return frame

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
