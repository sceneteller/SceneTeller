""" The base class of all things in BlenderProc. """

from typing import Any, Dict
import weakref

import bpy
import numpy as np
from mathutils import Vector, Euler, Color, Matrix, Quaternion

from blenderproc.python.utility.Utility import Utility, KeyFrame


class Struct:
    """
    The base class of all things in BlenderProc, this can be an Entity in the scene or a Material which is only applied
    to a MeshObject.
    """

    # Contains weak refs to all struct instances
    # As it only uses weak references, instances can still be removed by GC when all other references are gone.
    # If that happens, the instances' weak ref is also automatically removed from the set
    __refs__: weakref.WeakSet = weakref.WeakSet()

    def __init__(self, bpy_object: bpy.types.Object):
        self.blender_obj = bpy_object
        # Remember that this instance exists
        Struct.__refs__.add(self)

    def is_valid(self):
        """ Check whether the contained blender reference is valid.

        The reference might become invalid after an undo operation or when the referenced struct is deleted.

        :return: True, if it is valid.
        """
        return str(self.blender_obj) != "<bpy_struct, Object invalid>"

    def set_name(self, name: str):
        """ Sets the name of the struct.

        :param name: The new name.
        """
        self.blender_obj.name = name

    def get_name(self) -> str:
        """ Returns the name of the struct.

        :return: The name.
        """
        return self.blender_obj.name

    def get_cp(self, key: str, frame: int = None) -> Any:
        """ Returns the custom property with the given key.

        :param key: The key of the custom property.
        :param frame: The frame number which the value should be set to. If None is given, the current
                      frame number is used.
        :return: The value of the custom property.
        """
        with KeyFrame(frame):
            value = self.blender_obj[key]
            if isinstance(value, (Vector, Euler, Color, Matrix, Quaternion)):
                value = np.array(value)
            return value

    def set_cp(self, key: str, value: Any, frame: int = None):
        """ Sets the custom property with the given key. The key can not be the same as any member over the stored
        blender object.

        Keyframes can be only set for custom properties for the types int, float or bool.

        :param key: The key of the custom property.
        :param value: The value to set.
        :param frame: The frame number which the value should be set to. If None is given, the current
                      frame number is used.
        """
        if hasattr(self.blender_obj, key):
            raise ValueError(f"The given key: {key} is already an attribute of the blender object and can not be "
                             f"used as an custom property, please change the custom property name.")
        self.blender_obj[key] = value
        if isinstance(self.blender_obj[key], (float, int)):
            Utility.insert_keyframe(self.blender_obj, "[\"" + key + "\"]", frame)

    def del_cp(self, key: str):
        """ Removes the custom property with the given key.

        :param key: The key of the custom property to remove.
        """
        del self.blender_obj[key]

    def has_cp(self, key: str) -> bool:
        """ Return whether a custom property with the given key exists.

        :param key: The key of the custom property to check.
        :return: True, if the custom property exists.
        """
        return key in self.blender_obj

    def get_all_cps(self) -> Dict[str, Any]:
        """ Returns all custom properties as key, value pairs.

        :return: A dictionary of custom properties as key, value pairs
        """
        return dict(self.blender_obj.items())

    def clear_all_cps(self):
        """ Removes all existing custom properties the struct has. """
        # iterating over the keys is not possible as deleting them changes the structure of the
        # underlying blender object -> to solve this we always remove only the first element until no element is left
        while len(self.blender_obj.keys()) > 0:
            # extract first element of the keys
            key = list(self.blender_obj.keys())[0]
            # delete this first element
            del self.blender_obj[key]

    def get_attr(self, attr_name: str) -> Any:
        """ Returns the value of the attribute with the given name.

        :param attr_name: The name of the attribute.
        :return: The value of the attribute
        """
        if hasattr(self.blender_obj, attr_name):
            value = getattr(self.blender_obj, attr_name)
            if isinstance(value, (Vector, Euler, Color, Matrix, Quaternion)):
                value = np.array(value)
            return value
        raise ValueError(f"This element does not have an attribute {attr_name}")

    def __setattr__(self, key: str, value: Any):
        if key != "blender_obj":
            raise RuntimeError("The API class does not allow setting any attribute. Use the corresponding method or "
                               "directly access the blender attribute via entity.blender_obj.attribute_name")
        object.__setattr__(self, key, value)
