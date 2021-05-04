################################################################################
# Author:      Taisiya Parkhomenko
# MatNr:       01650051
# File:        config_reader.py
# Description: Reading the plotting configuration file.
# Comments:    -
################################################################################
import json


class ConfigReader:

    @staticmethod
    def read_json_config(config_file):  # static method. It reads the config file and returns the content as a dict
        with open(config_file) as a:
            config_info_dict = json.load(a)
            return config_info_dict
