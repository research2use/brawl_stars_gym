from datetime import datetime
from pathlib import Path
from re import sub

import cv2
import numpy as np
import pytesseract
from game_control.sprite import Sprite
from game_control.utilities import extract_roi_from_image

from brawl_stars_gym.brawl_stars import BrawlStars

"""
Extends BrawlStars game with event specific stuff,
like starting and stopping the event, determining the reward and determining
if the episode is done.

"""


class TryBrawler(BrawlStars):
    TRY_BRAWLER_DIR = Path("try_brawler")

    def __init__(self, episode_duration_in_seconds=10, brawler="Shelly", **kwargs):
        """Starts this Brawl Stars event; returns when event is started."""
        if brawler != "Shelly":
            raise NotImplementedError("Only Shelly implemented for now")
        print("episode_duration_in_seconds=", episode_duration_in_seconds)
        self._episode_duration_in_seconds = episode_duration_in_seconds

        super().__init__(**kwargs)

        self.sprites.update(
            Sprite.discover_sprites(
                Path(__file__).parent
                / self.DATA_DIR
                / self.TRY_BRAWLER_DIR
                / self.SPRITE_DIR
            )
        )

        self._regions.update(
            {
                "BUTTON_SHELLY": (100, 134, 304, 341),
                "BUTTON_TRY": (485, 44, 523, 234),
                "BUTTON_EXIT": (494, 520, 508, 543),
                # "BUTTON_PLAY": (459, 686, 516, 886),
                "REWARD_TRY_DAMAGE_PER_SECOND": (67, 848, 89, 900),
            }
        )

        self.start_event()

    def start_event(self):
        """Starts this Brawl Stars event; returns when event is started.

        Returns:
            Frame: First frame when started (or failed).

        Raises:
            RuntimeError: when event could not be started
        """
        region_names = ("BUTTON_BRAWLERS", "BUTTON_SHELLY", "BUTTON_TRY", "BUTTON_EXIT")
        clicks = (True, True, True, False)
        for region_name, click in zip(region_names, clicks):
            sprite = self.sprites["SPRITE_" + region_name]
            region = self.regions[region_name]
            found, frame, _ = self._wait_for_sprite(
                sprite, region=region, msg="Waiting for " + region_name
            )
            if not found:
                raise RuntimeError("Could not find sprite", sprite.name, "in time")

            if click:
                self.input_controller.click_screen_region(region)

        self._started_at = datetime.utcnow()

        return frame

    def stop_event(self):
        """Stops this Brawl Stars event; returns when in main screen.

        Returns:
            Frame: First frame when arrived at main screen
        """
        region_names = ("BUTTON_EXIT", "BUTTON_BACK", "BUTTON_BACK")
        clicks = (True, True, True)
        for region_name, click in zip(region_names, clicks):
            sprite = self.sprites["SPRITE_" + region_name]
            region = self.regions[region_name]
            found, frame, _ = self._wait_for_sprite(
                sprite, region=region, msg="Waiting for " + region_name
            )
            if not found:
                raise RuntimeError("Could not find sprite", sprite.name, "in time")

            if click:
                self.input_controller.click_screen_region(region)

        return frame

    def reset(self):
        """Restarts this Brawl Stars event; returns when event is restarted.

        Returns:
            Frame: First frame when event has started
        """
        region_names = ("BUTTON_EXIT", "BUTTON_TRY", "BUTTON_EXIT")
        clicks = (True, True, False)
        for region_name, click in zip(region_names, clicks):
            sprite = self.sprites["SPRITE_" + region_name]
            region = self.regions[region_name]
            found, frame, _ = self._wait_for_sprite(
                sprite, region=region, msg="Waiting for " + region_name
            )
            if not found:
                raise RuntimeError("Could not find sprite", sprite.name, "in time")

            if click:
                self.input_controller.click_screen_region(region)

        self._started_at = datetime.utcnow()

        return self.observation(frame)

    @staticmethod
    def _preprocess_text_image(img):
        """Converts image to a binary image where text is black on a white background.

        Args:
            img (np.ndarray): Image with reward text in BGR format;
                typically the region of interest of the full frame that contains the reward.

        Returns:
            np.ndarray: Preprocessed image with black reward text on white background.

        """
        text_color_bgr = np.array([255, 136, 136])
        text_color_bgr_dev = np.array([15] * 3)

        img = cv2.resize(img, (0, 0), fx=5, fy=5)
        img = cv2.inRange(
            img,
            text_color_bgr - text_color_bgr_dev,
            text_color_bgr + text_color_bgr_dev,
        )
        img = cv2.bitwise_not(img)
        img = cv2.blur(img, (2, 2))

        return img

    @staticmethod
    def damage_per_second(roi):
        """Extracts and returns the number in the given region of interest image.
        This number represents the damage per second that is displayed in this event.

        Args:
            roi (np.ndarray): The extracted region of interest of the full game frame
                that contains only the numbers to be extracted.

        Returns:
            Int: The extracted number representing the inflicted damage per second.

        """
        roi = TryBrawler._preprocess_text_image(roi)
        custom_config = r"--oem 1 --psm 6 outputbase digits"
        reward = pytesseract.image_to_string(roi, config=custom_config)
        reward = sub(r"\D", "", reward)

        return 0 if not reward else int(reward)

    def reward(self, frame):
        """Returns the reward of the previously performed action that resulted in the given Frame.

        The reward is defined as the total accumulated damage inflicted to enemies.
        This is approximated by the "damage per second" that is displayed in the top right corner
        of the game screen. Not perfect, but probably correlates nicely with the actual total
        damage and is far easier to extract than the the accumlated lost hitpoints of enemies.

        Args:
            frame (Frame): the resulting frame of the previously performed action.

        Returns:
            Number: The reward.

        """
        reward_roi = self.regions["REWARD_TRY_DAMAGE_PER_SECOND"]
        region = extract_roi_from_image(frame.img, reward_roi)

        return self.damage_per_second(region)

    def done(self, frame):
        """Returns if the episode has finshed, when its time has passed.

        Args:
            frame (Frame): the resulting frame of the previously performed action.
                Not used in this specific event, but required by generic interface.

        Returns:
            Bool: True when episode_duration_in_seconds that was given to constructor have passed;
                False otherwise.
        """
        current_episode_duration = datetime.utcnow() - self._started_at
        return (
            current_episode_duration.total_seconds()
            >= self._episode_duration_in_seconds
        )
