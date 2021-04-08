from datetime import datetime
from pathlib import Path

from game_control.sprite import Sprite

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

    def reward(self, frame):
        return 42

    def done(self, frame):
        current_episode_duration = datetime.utcnow() - self._started_at
        return (
            current_episode_duration.total_seconds()
            >= self._episode_duration_in_seconds
        )
