from pathlib import Path

from game_control.games.executable_game import ExecutableGame
from game_control.sprite import Sprite


class LDPlayer(ExecutableGame):
    LDPLAYER_DIR = Path("ldplayer")

    def __init__(self, ldplayer_executable_filepath, **kwargs):
        """
        Args:
            ldplayer_executable_filepath (string): Executable of LDPlayer
        """
        super().__init__(ldplayer_executable_filepath, window_name="LDPlayer", **kwargs)

        self.sprites.update(
            Sprite.discover_sprites(
                Path(__file__).parent
                / self.DATA_DIR
                / self.LDPLAYER_DIR
                / self.SPRITE_DIR
            )
        )

        self._wait_for_start()

    def _wait_for_start(self):
        """Wait for LDPlayer to fully start; returns when it is started.

        Returns:
            Frame: First frame when started (or failed).

        Raises:
            RuntimeError: when emulator could not be started
        """
        # LDPlayer sometimes resizes when fully started,
        # so search for sprite globally and re-initialize afterwards.
        sprite = self.sprites["SPRITE_LDPLAYER"]
        found, frame, location = self._wait_for_sprite(
            sprite, msg="Waiting for LDPlayer"
        )
        if not found:
            raise RuntimeError("Could not find sprite", sprite.name, "in time")

        self._initialize_window()
        print("Found sprite at location", location)

        return frame
