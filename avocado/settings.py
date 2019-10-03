"""Module to parse the settings file.

This file first reads the default settings file, and then optionally reads a
settings file in the working directory to override any of the settings.
"""
import json
import os

# Figure out the root directory for our package.
dirname = os.path.dirname
package_root_directory = dirname(dirname(os.path.abspath(__file__)))

# First, load the default settings
default_path = os.path.join(package_root_directory, "avocado_settings.json")
settings = json.load(open(default_path))

# Next, override with user settings
user_path = os.path.join(os.getcwd(), settings["user_settings_file"])
try:
    user_settings = json.load(open(user_path))
except FileNotFoundError:
    # No user settings available. Just use the defaults.
    pass
else:
    settings.update(user_settings)
