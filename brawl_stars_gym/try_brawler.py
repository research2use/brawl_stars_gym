from brawl_stars_gym.brawl_stars import BrawlStars

"""
Extends BrawlStars game with event specific stuff,
like starting and stopping the event, determining the reward and determining
if the episode is done.

"""


class TryBrawler(BrawlStars):
    def __init__(self, brawler="Shelly", **kwargs):
        """Starts this Brawl Stars event; returns when event is started."""
        if brawler != "Shelly":
            raise NotImplementedError("Only Shelly implemented for now")
        super().__init__(**kwargs)
        self.start_event()

    def start_event(self):
        """Starts this Brawl Stars event; returns when event is started.

        Returns:
            Frame: First frame when event has started
        """
        print("Starting try brawler event")
        return self.grab_frame()

    def stop_event(self):
        """Stops this Brawl Stars event; returns when in main screen.

        Returns:
            Frame: First frame when arrived at main screen
        """
        print("Stopping try brawler event")
        return self.grab_frame()

    def reward(self, frame):
        return 42

    def done(self, frame):
        return False
