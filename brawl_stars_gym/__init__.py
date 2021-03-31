"""Top-level package for brawl_stars_gym."""

__author__ = """Research 2 use"""
__email__ = "research2use@hotmail.com"
__version__ = "0.1.0"

from game_control.envs.game.game_env import GameEnv
from gym.envs.registration import register

from brawl_stars_gym.showdown_solo import ShowdownSolo
from brawl_stars_gym.try_brawler import TryBrawler

register(
    id="BrawlStarsTryBrawler-v0",
    entry_point="game_control.envs.game:GameEnv",
    kwargs={"game_class": TryBrawler},
)

register(
    id="BrawlStarsShowdownSolo-v0",
    entry_point="game_control.envs.game:GameEnv",
    kwargs={"game_class": ShowdownSolo},
)
