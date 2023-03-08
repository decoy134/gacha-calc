"""Module nikke_dmg for computing the DPS of various NIKKE combinations."""

import copy
import math
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
        if logger.hasHandlers():
            return logger

        logger.setLevel(logging.DEBUG)

        # Create handlers
        c_handler = logging.StreamHandler()
        c_handler.setLevel(logging.DEBUG)

        # Create formatters and add it to handlers
        #c_format = logging.Formatter('[%(name)s | %(levelname)s] (%(asctime)s) %(message)s')
        c_format = logging.Formatter('%(message)s')

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

    class ScatterPlot:
        """Create a scatter plot with a line through it."""
        def __init__(self, title: str):
            plt.style.use('classic')
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

        def draw_line(self, data: np.array, color: str='g'):
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
    # Table for elements that counter one another
    element_table = {
        'water': 'fire',
        'fire': 'wind',
        'wind': 'iron',
        'iron': 'electric',
        'electric': 'water'
    }

    # Emperical value for weapons
    weapon_table = {
        'AR': {
            'attack_speed': 60.0 / 5.0,
            'wind_up_seconds': 0.0,
            'wind_up_ammo': 0
        },
        'SMG': {
            'attack_speed': 225.0 / 12.0,
            'wind_up_seconds': 0.0,
            'wind_up_ammo': 0
        },
        'SR': {
            'attack_speed': 2.4,
            'wind_up_seconds': 0,
            'wind_up_ammo': 0
        },
        'MG': {
            'attack_speed': 52,
            'wind_up_seconds': 2,
            'wind_up_ammo': 45
        },
    }

    # Reference values for cubes, calls resilience 'reload' because it's easier to type...
    cube_table = {
        'reload': [0.0, 14.84, 22.27, 29.69],
        'bastion': [0.0, 0.1, 0.2, 0.3],
        'adjutant': [0.0, 1.06, 1.59, 2.12],
        'wingman': [0.0, 14.84, 22.27, 29.69],
        'onslaught': [0.0, 2.54, 3.81, 5.09],
        'assault': [0.0, 2.54, 3.81, 5.09],
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

        def get_normal_params(self, name: str) -> dict:
            """Returns a dictionary containing the relevant normal attack parameters."""
            return {
                "damage": self.get_normal_damage(name),
                "ammo": self.get_ammo_capacity(name),
                "reload": self.get_reload_seconds(name),
                "weapon": self.get_weapon_type(name),
            }

        def get_nikke_attack(self, name: str) -> float:
            """Returns a NIKKE's attack value from the configuration file."""
            return self.config['nikkes'][name]['attack']

        def get_enemy_defense(self, name: str) -> float:
            """Returns an enemy's defense value from the configuration file."""
            return self.config['enemies'][name]['defense']

        def get_weapon_type(self, name: str) -> str:
            """Returns the base ammo capacity of a specific Nikke."""
            return self.config['nikkes'][name]['weapon']

        def get_ammo_capacity(self, name: str) -> int:
            """Returns the base ammo capacity of a specific Nikke."""
            return self.config['nikkes'][name]['ammo']

        def get_normal_damage(self, name: str) -> float:
            """Returns the normal attack damage of a specific Nikke."""
            return self.config['nikkes'][name]['normal']

        def get_reload_seconds(self, name: str) -> float:
            """Returns the base ammo capacity of a specific Nikke."""
            return self.config['nikkes'][name]['reload']

        def add_skill_1(
                self,
                name: str,
                depth: int = None,
                start: float = 0.0,
                duration: float = None):
            """Adds any buffs from a NIKKE's Skill 1 to the buff list.
            
            Forwards the data to __pre_add_buff with the 'skill_1' key.
            """
            self.__pre_add_buff('skill_1', f'{name}_S1', name, depth, start, duration)

        def add_skill_2(
                self,
                name: str,
                depth: int = None,
                start: float = 0.0,
                duration: float = None):
            """Adds any buffs from a NIKKE's Skill 2 to the buff list.
            
            Forwards the data to __pre_add_buff with the 'skill_2' key.
            """
            self.__pre_add_buff('skill_2', f'{name}_S2', name, depth, start, duration)

        def add_burst(
                self,
                name: str,
                depth: int = None,
                start: float = 0.0,
                duration: float = None):
            """Adds any buffs from a NIKKE's Burst to the buff list.
            
            Forwards the data to __pre_add_buff with the 'burst' key.
            """
            self.__pre_add_buff('burst', f'{name}_B', name, depth, start, duration)


        @staticmethod
        def update_effect_duration(effect, start, duration):
            """Updates an effect by reference according to its start and duration."""
            if duration is not None:
                effect['duration'] = duration
            effect['start'] = start
            effect['end'] = start + effect.get('duration', math.inf)
            return effect

        def __pre_add_buff(
                self,
                skill_type: str,
                key: str,
                name: str,
                depth: int = None,
                start: float = 0.0,
                duration: float = None):
            """Adds any buffs from a NIKKE's skill key to the buff list.

            If the buff does not exist in the buff list, then it calls the next
            __add_buff utility to add it to the buff list.
            
            This function checks if the buff already exists in the buff list
            and whether it is stackable. If it is, it increases the buffs stack count.
            
            It also updates the buff's start, end, and duration metadata.
            """
            skill = self.config['nikkes'][name][skill_type]
            if key not in self.buffs:
                self.__add_buff(skill['effect'], key, depth, start, duration)
            elif isinstance(self.buffs[key], dict):
                effect = self.buffs[key]
                if skill['type'].startswith('stack'):
                    effect['stacks'] = effect.get('stacks', 1) + 1
                NIKKE.Config.update_effect_duration(effect, start, duration)

        def __add_buff(
                self,
                effect: list or dict,
                key: str,
                depth: int = None,
                start: float = 0.0,
                duration: float = None):
            """Adds a buff to the buff list, if the effect meets the requisite conditions.

            This function is called only when the buff given is not already present
            in the current buff list.
            """
            if isinstance(effect, list):
                length = len(effect)
                if isinstance(depth, int) and depth > 0 and depth <= length:
                    length = depth
                self.buffs[key] = []
                for i in range(length):
                    if effect[i]['type'] == 'buff':
                        self.buffs[key].append(NIKKE.Config.update_effect_duration(
                            copy.deepcopy(effect[i]), start, duration))
            elif effect['type'] == 'buff':
                self.buffs[key] = NIKKE.Config.update_effect_duration(
                    copy.deepcopy(effect), start, duration)

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
        core_hit: bool = False,
        range_bonus: bool = False,
        full_burst: bool = False,
        element_bonus: bool = False,

    @staticmethod
    def compute_normal_dps(
        damage: float,
        ammo: int,
        reload: float,
        weapon: str) -> float:
        """Computes the normal attack DPS of a character as multiplier/second.
        
        - damage: The damage multiplier of each normal attack.
        - ammo: The actual ammo of the character, after ammo percentage and bastion cube.
        - reload: The actual reload time of the character, after reload speed.
        - weapon: The key specifying what weapon this character uses.
        """
        speed = NIKKE.weapon_table[weapon]['attack_speed']
        wind_up_seconds = NIKKE.weapon_table[weapon]['wind_up_seconds']
        wind_up_ammo = NIKKE.weapon_table[weapon]['wind_up_ammo']
        return damage * ammo / ((ammo - wind_up_ammo) / speed + wind_up_seconds + reload)

    @staticmethod
    def compute_peak_normal_dps(damage: float, weapon: str) -> float:
        """Returns the maximum achievable multiplier/second for a 
        character's normal attack, in other words the infinite ammo case.
        """
        return damage * NIKKE.weapon_table[weapon]['attack_speed']

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
        # Total DEF (5)
        if buffs is None:
            buffs = []
        if cache is not None:
            calc = copy.deepcopy(cache)
            NIKKE.update_cache(buffs, calc)
        else:
            calc = NIKKE.generate_cache(buffs)
        calc.modifiers[0] = attack * calc.modifiers[0] / 100.0 - defense * calc.modifiers[5] / 100.0
        calc.modifiers[1] /= 100.0
        calc.modifiers[2] /= 100.0
        calc.modifiers[3] = 1.0 if not element_bonus and not calc.element_bonus \
            else calc.modifiers[3] / 100.0
        calc.modifiers[4] /= 100.0
        calc.modifiers[5] = 1.0

        base_mod = 1.0
        if core_hit or calc.core_hit:
            base_mod += 1.0
        if range_bonus or calc.range_bonus:
            base_mod += 0.3
        if full_burst or calc.full_burst:
            base_mod += 0.5

        crit_rate_p = calc.crit_rate / 100.0
        crit_dmg_p = calc.crit_dmg / 100.0
        crit_mod = base_mod + crit_dmg_p
        avg_mod = base_mod * (1.0 - crit_rate_p) + crit_mod * crit_rate_p

        final_atk = np.prod(calc.modifiers) * damage / 100.0
        return final_atk * np.array([base_mod, crit_mod, avg_mod])

    @staticmethod
    def generate_cache(buffs: list, crit_rate: float = 15, crit_dmg: float = 50) -> CachedModifiers:
        """Caches the modifier values and returns them in a dictionary.
        
        Use this function when looping to reduce the number of redundant
        computations from calling compute_damage() on a large buff list.
        """
        cache = NIKKE.CachedModifiers(
            np.array([100.0, 100.0, 100.0, 110.0, 100.0, 100.0]),
            crit_rate,crit_dmg, False, False, False, False)
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
            if 'full_charge_dmg' in buff:
                cache.modifiers[1] += (buff['full_charge_dmg'] - 100) * stacks
            if 'damage_taken' in buff:
                cache.modifiers[2] += buff['damage_taken'] * stacks
            if 'element_dmg' in buff:
                cache.modifiers[3] += buff['element_dmg'] * stacks
            if 'damage_up' in buff:
                cache.modifiers[4] *= 1.0 + buff['damage_up'] / 100.0 * stacks
            if 'defense' in buff:
                cache.modifiers[5] += buff['defense'] * stacks
            if 'crit_rate' in buff:
                cache.crit_rate += buff['crit_rate'] * stacks
            if 'crit_dmg' in buff:
                cache.crit_dmg += buff['crit_dmg'] * stacks
            # Bonus override flags
            if 'core_hit' in buff:
                cache.core_hit = buff['core_hit']
            if 'range_bonus' in buff:
                cache.range_bonus = buff['range_bonus']
            if 'full_burst' in buff:
                cache.full_burst = buff['full_burst']
            if 'element_bonus' in buff:
                cache.full_burst = buff['element_bonus']

    @staticmethod
    def get_bonus_tag(
            core_hit: bool = False,
            range_bonus: bool = False,
            full_burst: bool = False,
            element_bonus: bool = False) -> str:
        """Returns the tag corresponding to the flags."""
        tag = ''
        if core_hit:
            tag += 'core_'
        if range_bonus:
            tag += 'range_'
        if full_burst:
            tag += 'fb_'
        if element_bonus:
            tag += 'elem_'
        return tag[:-1] if tag != '' else 'base'

    @staticmethod
    def compute_damage_matrix(
            damage: float,
            attack: float,
            defense: float,
            buffs: list = None) -> dict:
        """Computes the matrix of damage dealt by source to target for
        all possibilities of core hit, range bonus, and
        
        Returns a 1x3 numpy array contain the no-crit, crit, and average damage.
        
        Currently does not take into account weakpoint damage due to confusion
        on how and when that triggers.
        """
        cache = NIKKE.generate_cache(buffs)
        ret = {
            'matrix': np.zeros((16,3))
        }
        index = 0
        flags = [False, True]
        for core_hit in flags:
            for range_bonus in flags:
                for full_burst in flags:
                    for element_bonus in flags:
                        tag = NIKKE.get_bonus_tag(
                            core_hit=core_hit,
                            range_bonus=range_bonus,
                            full_burst=full_burst,
                            element_bonus=element_bonus)
                        total_dmg = NIKKE.compute_damage(
                            damage,
                            attack,
                            defense,
                            core_hit=core_hit,
                            range_bonus=range_bonus,
                            full_burst=full_burst,
                            element_bonus=element_bonus,
                            cache=cache)
                        ret[tag] = {
                            'index': index,
                            'base': total_dmg[0],
                            'crit': total_dmg[1],
                            'avg': total_dmg[2]
                        }
                        ret['matrix'][index] = total_dmg

                        index += 1
        return ret

    @staticmethod
    def matrix_avg_dmg(matrix: dict, tags: dict, normalize: bool = True) -> float:
        """Sums damage from the damage matrix according to tags."""
        total_dmg = 0.0
        total_ratio = 0.0
        for tag, ratio in tags.items():
            total_dmg += matrix[tag]['avg'] * ratio
            total_ratio += ratio
        if normalize and total_ratio != 0:
            total_dmg /= total_ratio
        return total_dmg

    @staticmethod
    def accumulate_avg_dmg(
            damage: float,
            attack: float,
            defense: float,
            buffs: list,
            tags: dict,
            normalize: bool = True) -> float:
        """Sums damage from the damage matrix according to tags."""
        matrix = NIKKE.compute_damage_matrix(damage, attack, defense, buffs=buffs)
        return NIKKE.matrix_avg_dmg(matrix, tags, normalize=normalize)

    @staticmethod
    def compute_dps_window(
            damage_tags: list,
            attack: float,
            defense: float,
            buffs: list,
            window_start: float = -math.inf,
            window_end: float = math.inf,
            accumulate: bool = True,
            normalize: bool = True) -> float:
        """Uses start and end times in the buff iist to estimate damage.
        
        damage_tags specifies the damage to sum over and the associated
        bonuses from core hits, full burst, etc., and in what duration
        window they apply.

        normalize is passed to the accumulate function.

        accumulate when True will sum the final result as a single float.

        window_start manually specifies the minimum time to analyaze from.

        window_end manually specifies the maximum time to analyze towards.
        """
        # Start by searching through the buff list to determine timeline
        time_points = np.array([])
        if not math.isinf(window_start):
            time_points = np.append(time_points, window_start)
        if not math.isinf(window_end):
            time_points = np.append(time_points, window_end)
        for buff in buffs:
            start = buff['start']
            end = buff['end']
            if not math.isinf(start) \
                    and window_start < start < window_end \
                    and start not in time_points:
                time_points = np.append(time_points, start)
            if not math.isinf(end) \
                    and window_start < end < window_end \
                    and end not in time_points:
                time_points = np.append(time_points, end)

        # Sort the timeline in chronological order
        time_points = np.sort(time_points)

        # Preinitialize the list of time point windows
        buff_windows = []
        t_0 = time_points[0]
        for t_1 in time_points:
            if t_0 >= t_1:
                continue
            window = []
            for buff in buffs:
                if t_0 >= buff['start'] and t_0 < buff['end']:
                    window.append(buff)
            buff_windows.append((t_0, t_1, window))
            t_0 = t_1

        # Loop over all damage tags and begin accumulating the damage per window
        results = np.zeros(len(damage_tags))
        for i, dmg_tag in enumerate(damage_tags):
            damage = dmg_tag['damage']
            tags = dmg_tag['tags']
            start = dmg_tag['start']
            duration = dmg_tag.get('duration', 0)
            end = dmg_tag.get('end', start + duration if not math.isinf(duration) else math.inf)

            # Timeline loop - t0 is the inclusive current time and t1 is non-inclusive the end time
            for t_0, t_1, window in buff_windows:
                # Shift the window until we are in a valid start time
                if start >= t_1:
                    continue
                # Break if the window start exceeds the end time
                if end < t_0:
                    break

                total_dmg = NIKKE.accumulate_avg_dmg(
                    damage,
                    attack,
                    defense,
                    window,
                    tags,
                    normalize
                )

                # Determine, based on duration, whether or not to multiply
                duration = min(end, t_1) - max(start, t_0) if duration != 0 else 0
                if duration > 0:
                    results[i] += total_dmg * duration
                else:
                    results[i] += total_dmg

                start = t_1
        return np.sum(results) if accumulate else results


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


class Helpers:
    """Namespace for helper functions specific to this script."""
    @staticmethod
    def compute_normal_attack_dps(
            config: NIKKE.Config,
            nikke_name: str,
            damage: float = None,
            ammo: float = 0.0,
            reload: float = NIKKE.cube_table['reload'][2],
            log: bool = True,
            graph: bool = False,
            atk_name: str = 'Normal Attack') -> float:
        """Graphs the normal attack DPS for the given NIKKE and logs it."""
        params = config.get_normal_params(nikke_name)
        if damage is not None:
            params['damage'] = damage
        base_ammo = params['ammo']
        params['ammo'] = int(base_ammo * (1 + ammo / 100))
        params['reload'] *= (1 - reload / 100)
        dps = NIKKE.compute_normal_dps(**params)
        peak = NIKKE.compute_peak_normal_dps(params['damage'], params['weapon'])
        ratio = dps / peak * 100
        message = f'{nikke_name} {atk_name} DPS: {dps:,.2f} / {peak:,.2f} ({ratio:,.2f}%)'

        if log:
            Util.get_logger('NIKKE_Logger').info(message)

        # Add a plot for this graph
        if graph:
            iterations = 25
            data = np.zeros((iterations, 2))
            for i in range(iterations):
                data[i][0] = 1 + 0.1 * i + ammo / 100
                params['ammo'] = int(base_ammo * data[i][0])
                data[i][0] = (data[i][0] - 1) * 100
                data[i][1] = ((NIKKE.compute_normal_dps(**params) / dps) - 1) * 100
            plot = Graphs.ScatterPlot(f'{nikke_name} {atk_name} DPS vs Ammo (Lv2 Reload Cube)')
            plot.draw_line(data)
            plot.set_xlabel('Ammo Capacity Up (%)')
            plot.set_ylabel('Damage Increase (%)')
            plt.show()

        return dps

    @staticmethod
    def compute_actual_damage(
            damage: float,
            attack: float,
            defense: float,
            buffs: list,
            log: bool = True) -> np.array:
        """Returns the damage a single hit does under burst and logs it."""
        dmg_cache = NIKKE.generate_cache(buffs)
        values = NIKKE.compute_damage(damage, attack, defense, cache=dmg_cache)
        if log:
            logger = Util.get_logger('NIKKE_Logger')
            msg = f'Scarlet burst damage based on the following stats:\
                \n  - ATK: {attack}\
                \n  - Enemy DEF: {defense}\
                \n  - Skill Multiplier: {damage:.2f}'
            logger.info(msg)
            logger.info('Base Damage: %s', f'{values[0]:,.2f}')
            logger.info('Crit Damage: %s', f'{values[1]:,.2f}')
            logger.info('Average Damage: %s', f'{values[2]:,.2f}')
        return values

    @staticmethod
    def compute_nikke_dps(
            damage_tags: list,
            attack: float,
            defense: float,
            buffs: list,
            window_start: float,
            window_end: float,
            name: str = 'NIKKE',
            relative_dps: float = None,
            relative_name: str = None,
            verbose=False) -> float:
        """Returns the average DPS of a NIKKE."""
        logger = Util.get_logger('NIKKE_Logger')
        total_avg_dmg = NIKKE.compute_dps_window(
            damage_tags=damage_tags,
            attack=attack,
            defense=defense,
            buffs=buffs,
            window_start=window_start,
            window_end=window_end)
        dps = total_avg_dmg / (window_end - window_start)

        if verbose:
            duration = window_end - window_start
            msg = f'{name} Average DPS based on the following stats:\
                \n  - ATK: {attack}\
                \n  - Enemy DEF: {defense}\
                \n  - Duration: {duration:.2f}'
            logger.debug(msg)

        msg = f'{name} Average Damage = {total_avg_dmg:,.2f} ({dps:,.2f} damage/s)'
        if relative_dps is not None:
            ratio = dps / relative_dps * 100.0
            msg += f' ({ratio:,.2f}% of {relative_name})' \
                if relative_name is not None else f' ({ratio:,.2f}%)'
        logger.info(msg)
        return dps


def main() -> int:
    """Main function."""
    logger = Util.get_logger('NIKKE_Logger')
    config = NIKKE.Config()
    params = {
        'damage': config.config['nikkes']['Scarlet']['burst']['effect'][1]['damage'],
        'attack': config.get_nikke_attack('Modernia'),
        'defense': config.get_enemy_defense('special_interception'),
    }

    # Base buffs
    ammo = 152.50
    burst_times = [0, 13, 26]
    base_buffs = []
    for start in burst_times:
        config.add_skill_1('Liter', start=start-0.5, depth=3)
        config.add_burst('Liter', start=start-0.5)
        #config.add_burst('Novel', start=start+1.5)
        base_buffs += config.get_buff_list()
        base_buffs.append({
            'full_burst': True,
            'start': start,
            'end': start + 10,
            'duration': 10
        })
        config.clear_buffs()

    # Scarlet buffs
    for _ in range(5):
        config.add_skill_1('Scarlet', duration=math.inf)
    config.add_skill_2('Scarlet', duration=math.inf)
    config.add_burst('Scarlet', start=burst_times[0])
    scar_buffs = base_buffs + config.get_buff_list()
    config.clear_buffs()

    # Modernia buffs
    for _ in range(5):
        config.add_skill_1('Modernia', duration=math.inf)
    config.add_skill_2('Modernia', duration=math.inf)
    mod_buffs = base_buffs + config.get_buff_list()
    logger.debug(mod_buffs)

    logger.info('=======================================================')
    Helpers.compute_actual_damage(**params, buffs=scar_buffs)
    logger.info('=======================================================')
    scar_n = Helpers.compute_normal_attack_dps(
        config, 'Scarlet', ammo=ammo, graph=False)
    sw_n = Helpers.compute_normal_attack_dps(
        config, 'Snow White', ammo=0, graph=False)
    mod_n = Helpers.compute_normal_attack_dps(
        config, 'Modernia', graph=False,
        ammo=config.config['nikkes']['Modernia']['skill_1']['effect']['ammo']*5+ammo)
    mod_s1 = Helpers.compute_normal_attack_dps(
        config, 'Modernia', graph=False, atk_name='S1',
        damage=config.config['nikkes']['Modernia']['skill_1']['effect']['damage'],
        ammo=config.config['nikkes']['Modernia']['skill_1']['effect']['ammo']*5+ammo)
    max_n = Helpers.compute_normal_attack_dps(
        config, 'Maxwell', ammo=ammo, graph=False)
    alice_n = Helpers.compute_normal_attack_dps(
        config, 'Alice', ammo=ammo, graph=False)
    logger.info('=======================================================')

    # Set up weapon tags
    ar_range_bonus = False
    mg_range_bonus = False
    sr_range_bonus = False
    core_hit = False
    ar_base_tag = NIKKE.get_bonus_tag(range_bonus=ar_range_bonus)
    ar_core_tag = NIKKE.get_bonus_tag(range_bonus=ar_range_bonus, core_hit=core_hit)
    mg_core_tag = NIKKE.get_bonus_tag(range_bonus=mg_range_bonus, core_hit=core_hit)
    sr_base_tag = NIKKE.get_bonus_tag(range_bonus=sr_range_bonus)
    sr_core_tag = NIKKE.get_bonus_tag(range_bonus=sr_range_bonus, core_hit=core_hit)
    mod_s1_tag = NIKKE.get_bonus_tag(range_bonus=False)
    ar_tag_profile = {
        ar_base_tag: 0.8,
        ar_core_tag: 0.2
    }
    mg_tag_profile = {mg_core_tag: 1.0}
    sr_tag_profile = {
        sr_base_tag: 0.5,
        sr_core_tag: 0.5,
    }

    # Scarlet attack dps calculation
    scar_avg_dps = Helpers.compute_nikke_dps(
        damage_tags=[
            {
                'damage': config.config['nikkes']['Scarlet']['burst']['effect'][1]['damage'],
                'start': burst_times[-1] - 0.1,
                'duration': 0,
                'tags': {'base': 1.0},
            },
            {
                'damage': scar_n,
                'start': -math.inf,
                'duration': math.inf,
                'tags': ar_tag_profile,
            },
        ],
        attack=config.get_nikke_attack('Modernia'),
        defense=config.get_enemy_defense('special_interception'),
        buffs=scar_buffs,
        window_start=0,
        window_end=burst_times[-1],
        name='Scarlet (Self Burst)',
        verbose=False
    )

    # Modernia attack dps calculation
    Helpers.compute_nikke_dps(
        damage_tags=[
            {
                'damage': mod_n,
                'start': -math.inf,
                'duration': math.inf,
                'tags': mg_tag_profile,
            },
            {
                'damage': mod_s1,
                'start': -math.inf,
                'duration': math.inf,
                'tags': {mod_s1_tag: 1.0},
            },
        ],
        attack=config.get_nikke_attack('Modernia'),
        defense=config.get_enemy_defense('special_interception'),
        buffs=mod_buffs,
        window_start=0,
        window_end=burst_times[-1],
        name='Modernia',
        relative_dps=scar_avg_dps,
        relative_name='Scarlet'
    )

    # Snow White, Maxwell, and Alice Calculation
    mx_buffs = base_buffs
    for start in burst_times:
        config.add_skill_1('Maxwell', start=start)
        mx_buffs = mx_buffs + config.get_buff_list()
        config.clear_buffs()

    alice_dmg_tags = [
        {
            'damage': alice_n * 3.57,
            'start': burst_times[0],
            'duration': 10,
            'tags': sr_tag_profile
        },
        {
            'damage': alice_n * (1 + 2.5 / (NIKKE.weapon_table['SR']['attack_speed'] * 1.5)),
            'start': burst_times[0] + 10,
            'duration': math.inf,
            'tags': sr_tag_profile
        }
    ]
    config.add_burst('Alice', start=burst_times[0])
    alice_buffs = mx_buffs + config.get_buff_list()
    config.clear_buffs()

    reload_t = 1.5*(1 - NIKKE.cube_table['reload'][2] / 100.0)
    b_fire_t = burst_times[0] + 4.0
    s1_trigger_req = 30 / NIKKE.weapon_table['AR']['attack_speed']
    restart_t = b_fire_t + reload_t
    s1_trigger_t = restart_t + s1_trigger_req
    config.add_skill_2('Snow White', start=burst_times[0]+2)
    config.add_skill_1('Snow White', start=s1_trigger_t, duration=math.inf)
    sw_buffs = mx_buffs + config.get_buff_list()
    config.clear_buffs()
    config.add_skill_2('Snow White', start=burst_times[0]+17)
    sw_buffs = sw_buffs + config.get_buff_list()

    sw_dmg_tags = [
        {
            'damage': 499.5 * 9,
            'start': b_fire_t,
            'duration': 0,
            'tags': {ar_core_tag: 1.0},
        },
        {
            'damage': sw_n,
            'start': restart_t,
            'duration': math.inf,
            'tags': ar_tag_profile,
        },
        {
            'damage': 126.64,
            'start': 2.0,
            'duration': 0,
            'tags': {'base': 1.0},
        },
        {
            'damage': 126.64,
            'start': 17.0,
            'duration': 0,
            'tags': {'base': 1.0},
        }
    ]
    for i in range(10):
        sw_dmg_tags.append({
            'damage': 65.55,
            'start': s1_trigger_t + s1_trigger_req * i,
            'duration': 0,
            'tags': {'base': 1.0},
        })
    Helpers.compute_nikke_dps(
        damage_tags=sw_dmg_tags,
        attack=config.get_nikke_attack('Snow White'),
        defense=config.get_enemy_defense('special_interception'),
        buffs=sw_buffs,
        window_start=0,
        window_end=burst_times[-1],
        name='Snow White',
        relative_dps=scar_avg_dps,
        relative_name='Scarlet',
        verbose=False
    )
    Helpers.compute_nikke_dps(
        damage_tags=[
            {
                'damage': 813.42 * 3,
                'start': 2.0,
                'duration': 0,
                'tags': {sr_core_tag: 1.0},
            },
            {
                'damage': max_n * (1 + 1.5 / (NIKKE.weapon_table['SR']['attack_speed'] * 1)),
                'start': 2.5,
                'duration': math.inf,
                'tags': sr_tag_profile,
            },
        ],
        attack=config.get_nikke_attack('Maxwell'),
        defense=config.get_enemy_defense('special_interception'),
        buffs=mx_buffs,
        window_start=0,
        window_end=burst_times[-1],
        name='Maxwell',
        relative_dps=scar_avg_dps,
        relative_name='Scarlet',
        verbose=False
    )
    Helpers.compute_nikke_dps(
        damage_tags=alice_dmg_tags,
        attack=config.get_nikke_attack('Maxwell'),
        defense=config.get_enemy_defense('special_interception'),
        buffs=alice_buffs,
        window_start=0,
        window_end=burst_times[-1],
        name='Alice',
        relative_dps=scar_avg_dps,
        relative_name='Scarlet',
        verbose=False
    )


    # Modernia burst attack dps calculation
    burst_times = [0, 18, 31]
    base_buffs = []
    for start in burst_times:
        config.add_skill_1('Liter', start=start-0.5, depth=3)
        config.add_burst('Liter', start=start-0.5)
        config.add_burst('Novel', start=start+1.5)
        base_buffs += config.get_buff_list()
        duration = 15 if start == burst_times[0] else 10
        base_buffs.append({
            'full_burst': True,
            'start': start,
            'end': start + duration,
            'duration': duration
        })
        config.clear_buffs()

    # Scarlet buffs
    for _ in range(5):
        config.add_skill_1('Scarlet', duration=math.inf)
    config.add_skill_2('Scarlet', duration=math.inf)
    config.add_burst('Scarlet', start=burst_times[1])
    scar_buffs = base_buffs + config.get_buff_list()
    config.clear_buffs()

    # Modernia buffs
    for _ in range(5):
        config.add_skill_1('Modernia', duration=math.inf)
    config.add_skill_2('Modernia', duration=math.inf)
    mod_buffs = base_buffs + config.get_buff_list()

    logger.info('=======================================================')
    Helpers.compute_nikke_dps(
        damage_tags=[
            {
                'damage': scar_n,
                'start': -math.inf,
                'duration': math.inf,
                'tags': ar_tag_profile,
            },
            {
                'damage': config.config['nikkes']['Scarlet']['burst']['effect'][1]['damage'],
                'start': burst_times[-1] - 0.1,
                'duration': 0,
                'tags': {'base': 1.0},
            },
        ],
        attack=config.get_nikke_attack('Modernia'),
        defense=config.get_enemy_defense('special_interception'),
        buffs=scar_buffs,
        window_start=0,
        window_end=burst_times[-1],
        name='Scarlet (Modernia Burst)',
        relative_dps=scar_avg_dps,
        relative_name='Normal',
        verbose=False
    )
    mod_b = 2.24 * 2 * NIKKE.weapon_table['MG']['attack_speed']
    mod_s1_b = 3.05 * NIKKE.weapon_table['MG']['attack_speed']
    Helpers.compute_nikke_dps(
        damage_tags=[
            {
                'damage': mod_b,
                'start': 0.0,
                'duration': math.inf,
                'tags': mg_tag_profile,
            },
            {
                'damage': mod_s1_b,
                'start': 0.0,
                'duration': math.inf,
                'tags': {mod_s1_tag: 1.0},
            },
        ],
        attack=config.get_nikke_attack('Modernia'),
        defense=config.get_enemy_defense('special_interception'),
        buffs=mod_buffs,
        window_start=0,
        window_end=burst_times[-1],
        name='Modernia Burst',
        relative_dps=scar_avg_dps,
        relative_name='Normal Scarlet',
        verbose=False
    )

    return 0

if __name__ == '__main__':
    main()
