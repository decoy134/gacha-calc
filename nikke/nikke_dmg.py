"""Module nikke_dmg for computing the DPS of various NIKKE combinations."""

import copy
import logging
import json
import os
from dataclasses import dataclass

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

class Util:
    """Utility namespace for static functions."""
    @staticmethod
    def get_default_config() -> str:
        """Loads the default configuration for this script."""
        this_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(this_dir, 'config', 'nikke.json')

    @staticmethod
    def get_logger(name: str, log_file: str = None) -> logging.Logger:
        """Returns a logger using the specified name."""
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)

        # Create handlers
        c_handler = logging.StreamHandler()
        c_handler.setLevel(logging.DEBUG)

        # Create formatters and add it to handlers
        c_format = logging.Formatter('[%(name)s | %(levelname)s] (%(asctime)s) %(message)s')
        c_handler.setFormatter(c_format)

        # Add handlers to the logger
        logger.addHandler(c_handler)

        if log_file is not None:
            # Create handlers
            f_handler = logging.FileHandler(log_file)
            f_handler.setLevel(logging.ERROR)

            # Create formatters and add it to handlers
            f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            f_handler.setFormatter(f_format)

            # Add handlers to the logger
            logger.addHandler(f_handler)

        return logger


class Graphs:
    """Graphing namespace for creating charts."""
    class ColorPlot:
        """Create a color plot with a 2D axis."""
        def __init__(self, title: str):
            plt.style.use('dark_background')
            self.fig, self.axes = plt.subplots(nrows=1, ncols=1)
            self.axes.minorticks_on()
            self.axes.grid(visible=True,
                which='major',
                linestyle='-',
                linewidth='0.5',
                color='red')
            self.axes.grid(visible=True,
                which='minor',
                linestyle=':',
                linewidth='0.5',
                color='black')
            self.set_title(title)
            self.set_xlabel('X-Axis')
            self.set_ylabel('Y-Axis')
            self.extent = None
            self.pos = None

        def load_data(self, data: np.array, style: str='turbo'):
            """Plots the specified data table as a color map, using specified color map style.
            
            'data' must be a MxN numpy array of floats for this function to plot.
            """
            color_scheme = matplotlib.colormaps[style]
            self.pos = self.axes.imshow(data,
                interpolation='none',\
                cmap=color_scheme,
                origin='lower',
                extent=self.extent,
                aspect='auto')
            self.fig.colorbar(self.pos, ax=self.axes)

        def draw_line(self, data: np.array, color: str='r'):
            """Draws a line in the color plot, using a Nx2 numpy array."""
            self.axes.plot(data[:,0], data[:,1], color=color)

        def set_title(self, title: str):
            """Sets the title of this plot."""
            self.title = title
            self.axes.set_title(self.title)

        def set_xlabel(self, label: str):
            """Sets the name of the x-axis for this plot."""
            self.axes.set_xlabel(label)

        def set_ylabel(self, label: str):
            """Sets the name of the y-axis for this plot."""
            self.axes.set_ylabel(label)

        def set_bounds(self, min_x: float, max_x: float, min_y: float, max_y: float):
            """Sets the maximum value of the x-axis for this plot."""
            self.extent = [min_x, max_x, min_y, max_y]

    class Histogram:
        """Create a histogram."""
        def __init__(self):
            plt.style.use('dark_background')

        def abc(self):
            """abc"""


class NIKKE:
    """API for computing various NIKKE calculations."""
    element_table = {
        'water': 'fire',
        'fire': 'wind',
        'wind': 'iron',
        'iron': 'electric',
        'electric': 'water'
    }

    weapon_table = {
        'AR': 60.0 / 5.0,
        'SMG': 225.0 / 12.0,
        'MG': 300 / 7.0,
    }

    class Config:
        """Manager object for searching the NIKKE JSON configuration file."""
        def __init__(self, fname: str = Util.get_default_config()):
            with open(fname, 'r', encoding='utf-8') as config_file:
                self.config = json.load(config_file)
            self.buffs = {}

        def get_buff_list(self) -> list:
            """Flattens the buff list for damage calculation parsing."""
            ret = []
            for item in self.buffs.values():
                if isinstance(item, list):
                    ret += item
                else:
                    ret.append(item)
            return ret

        def get_nikke_attack(self, name: str) -> float:
            """Returns a NIKKE's attack value from the configuration file."""
            return self.config['nikkes'][name]['attack']

        def get_enemy_defense(self, name: str) -> float:
            """Returns an enemy's defense value from the configuration file."""
            return self.config['enemies'][name]['defense']

        def add_skill_1(self, name: str, depth: int = None):
            """Adds any buffs from a NIKKE's Skill 1 to the buff list."""
            skill = self.config['nikkes'][name]['skill_1']
            key = f'{name}_S1'
            if key not in self.buffs:
                self.add_buff(skill['effect'], key, depth)
            elif skill['type'].startswith('stack'):
                self.buffs[key]['stacks'] = self.buffs[key].get('stacks', 1) + 1

        def add_skill_2(self, name: str, depth: int = None):
            """Adds any buffs from a NIKKE's Skill 2 to the buff list."""
            skill = self.config['nikkes'][name]['skill_2']
            key = f'{name}_S2'
            if key not in self.buffs:
                self.add_buff(skill['effect'], key, depth)
            elif skill['type'].startswith('stack'):
                self.buffs[key]['stacks'] = self.buffs[key].get('stacks', 0) + 1

        def add_burst(self, name: str, depth: int = None):
            """Adds any buffs from a NIKKE's Burst to the buff list."""
            skill = self.config['nikkes'][name]['burst']
            key = f'{name}_B'
            if key not in self.buffs:
                self.add_buff(skill['effect'], key, depth)
            elif skill['type'].startswith('stack'):
                self.buffs[key]['stacks'] = self.buffs[key].get('stacks', 0) + 1

        def add_buff(self, effect: list or dict, key: str, depth: int = None):
            """Adds a buff to the buff list, if the effect meets the requisite conditions."""
            if isinstance(effect, list):
                length = len(effect)
                if isinstance(depth, int) and depth > 0 and depth <= length:
                    length = depth
                self.buffs[key] = []
                for i in range(length):
                    if effect[i]['type'] == 'buff':
                        self.buffs[key].append(effect[i])
            elif effect['type'] == 'buff':
                self.buffs[key] = effect

        def clear_buffs(self):
            """Empties the internal buff list."""
            self.buffs = {}


    class Exceptions:
        """Exceptions namespace for NIKKE calculator."""
        class BadElement(Exception):
            """Signifies that a non-existent element was used for comparison."""


    @dataclass
    class CachedModifiers:
        """A data class for caching calculations which do not remove buffs from the buff list."""
        modifiers: np.array
        crit_rate: float = 15
        crit_dmg: float = 50


    @staticmethod
    def compute_damage(
            damage: float,
            attack: float,
            defense: float,
            buffs: list = None,
            core_hit: bool = False,
            range_bonus: bool = False,
            full_burst: bool = False,
            element_bonus: bool = False,
            cache: CachedModifiers = None) -> np.array:
        """Computes the damage dealt by source to target.
        
        Returns a 1x3 numpy array contain the no-crit, crit, and average damage.
        
        Currently does not take into account weakpoint damage due to confusion
        on how and when that triggers.
        """
        # Total ATK (0)
        # Charge Damage (1)
        # Damage Taken (2)
        # Elemental Damage (3)
        # Unique Modifiers (4)
        if buffs is None:
            buffs = []
        if cache is not None:
            calc = copy.deepcopy(cache)
            NIKKE.update_cache(buffs, calc)
        else:
            calc = NIKKE.generate_cache(buffs)
        calc.modifiers[0] =  attack * calc.modifiers[0] / 100.0 - defense
        calc.modifiers[1] /= 100.0
        calc.modifiers[2] /= 100.0
        calc.modifiers[3] = 1.0 if not element_bonus else calc.modifiers[3] / 100.0
        calc.modifiers[4] /= 100.0

        base_mod = 1.0
        if core_hit:
            base_mod += 0.5
        if range_bonus:
            base_mod += 0.3
        if full_burst:
            base_mod += 0.5

        crit_rate_p = calc.crit_rate / 100.0
        crit_dmg_p = calc.crit_dmg / 100.0
        crit_mod = base_mod + crit_dmg_p
        avg_mod = base_mod * (1.0 - crit_rate_p) + crit_mod * crit_rate_p

        final_atk = np.prod(calc.modifiers) * damage / 100.0
        if element_bonus:
            final_atk *= 1.1
        return final_atk * np.array([base_mod, crit_mod, avg_mod])

    @staticmethod
    def generate_cache(buffs: list, crit_rate: float = 15, crit_dmg: float = 50) -> CachedModifiers:
        """Caches the modifier values and returns them in a dictionary.
        
        Use this function when looping to reduce the number of redundant
        computations from calling compute_damage() on a large buff list.
        """
        cache = NIKKE.CachedModifiers(
            np.array([100.0, 100.0, 100.0, 110.0, 100.0]),
            crit_rate,
            crit_dmg)
        NIKKE.update_cache(buffs, cache)
        return cache

    @staticmethod
    def update_cache(buffs: list, cache: CachedModifiers):
        """Updates a modifier cache by reference."""
        for buff in buffs:
            stacks = int(buff.get('stacks', 1))
            if 'attack' in buff:
                cache.modifiers[0] += buff['attack'] * stacks
            if 'charge_dmg' in buff:
                cache.modifiers[1] += buff['charge_dmg'] * stacks
            if 'damage_taken' in buff:
                cache.modifiers[2] += buff['damage_taken'] * stacks
            if 'element_dmg' in buff:
                cache.modifiers[3] += buff['element_dmg'] * stacks
            if 'damage_up' in buff:
                cache.modifiers[4] *= 1.0 + buff['damage_up'] / 100.0 * stacks
            if 'crit_rate' in buff:
                cache.crit_rate += buff['crit_rate'] * stacks
            if 'crit_dmg' in buff:
                cache.crit_dmg += buff['crit_dmg'] * stacks


    @staticmethod
    def compare_element(source: str, target: str) -> bool:
        """Checks if the source element is strong against the target element.
        
        Returns True if the source element is strong against the target. False otherwise.
        """
        if not source in NIKKE.element_table:
            raise NIKKE.Exceptions.BadElement(f'"{source}" is not an element.')
        if not target in NIKKE.element_table:
            raise NIKKE.Exceptions.BadElement(f'"{target}" is not an element.')
        return NIKKE.element_table[source] == target



def main() -> int:
    """Main function."""
    logger = Util.get_logger('NIKKE_Logger')
    config = NIKKE.Config()
    params = {
        'damage':  config.config['nikkes']['Scarlet']['burst']['effect'][1]['damage'],
        'attack': config.get_nikke_attack('Scarlet'),
        'defense': config.get_enemy_defense('shooting_range'),
    }
    config.add_skill_1('Scarlet')
    config.add_skill_1('Scarlet')
    config.add_skill_1('Scarlet')
    config.add_skill_1('Scarlet')
    config.add_skill_1('Liter', depth=3)
    config.add_burst('Liter')
    config.add_burst('Scarlet')

    buffs = config.get_buff_list()
    logger.debug(buffs)

    dmg_cache = NIKKE.generate_cache(buffs)
    values = NIKKE.compute_damage(**params, cache=dmg_cache)

    logger.info('Scarlet burst damage based on the following stats:\
                \n  - ATK: %d\
                \n  - Enemy DEF: %d\
                \n  - Skill Multiplier: %.2f',
                int(params['attack']), int(params['defense']), params['damage'])
    logger.info('Base Damage: %s', f'{values[0]:,.2f}')
    logger.info('Crit Damage: %s', f'{values[1]:,.2f}')
    logger.info('Average Damage: %s', f'{values[2]:,.2f}')

    # Make a graph for fun
    data = np.zeros(shape=(5, 5))
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            data[i][j] = 1 + i * 0.1 + j * 0.1

    color_plot = Graphs.ColorPlot('Test Plot')
    color_plot.load_data(data)
    plt.show()

    return 0

if __name__ == '__main__':
    main()
