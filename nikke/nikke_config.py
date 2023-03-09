"""Config class for NIKKE API."""
import copy
import json
import logging
import math
import os

class NIKKEUtil:
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

class NIKKEConfig:
    """Manager object for searching the NIKKE JSON configuration file."""
    def __init__(self, fname: str = NIKKEUtil.get_default_config()):
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
            NIKKEConfig.update_effect_duration(effect, start, duration)

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
                    self.buffs[key].append(NIKKEConfig.update_effect_duration(
                        copy.deepcopy(effect[i]), start, duration))
        elif effect['type'] == 'buff':
            self.buffs[key] = NIKKEConfig.update_effect_duration(
                copy.deepcopy(effect), start, duration)

    def clear_buffs(self):
        """Empties the internal buff list."""
        self.buffs = {}

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
