"""Gait pattern planning module."""

# from __future__ import google_type_annotations
from __future__ import absolute_import, division, print_function

import abc
import enum


class LegState(enum.Enum):
    """The state of a leg during locomotion."""

    SWING = 0
    STANCE = 1
    # A swing leg that collides with the ground.
    EARLY_CONTACT = 2
    # A stance leg that loses contact.
    LOSE_CONTACT = 3


class GaitGenerator(object, metaclass=abc.ABCMeta):  # pytype: disable=ignored-metaclass
    """Generates the leg swing/stance pattern for the robot."""

    @abc.abstractmethod
    def reset(self):
        pass

    @abc.abstractmethod
    def update(self):
        pass
