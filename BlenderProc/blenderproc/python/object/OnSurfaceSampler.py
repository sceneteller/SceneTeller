"""Sampling objects on a surface."""

from typing import Callable, List, Optional, Dict

import mathutils
import numpy as np

from blenderproc.python.utility.CollisionUtility import CollisionUtility
from blenderproc.python.types.MeshObjectUtility import MeshObject


def sample_poses_on_surface(objects_to_sample: List[MeshObject], surface: MeshObject,
                            sample_pose_func: Callable[[MeshObject], None], max_tries: int = 100,
                            min_distance: float = 0.25, max_distance: float = 0.6,
                            up_direction: Optional[np.ndarray] = None,
                            check_all_bb_corners_over_surface: bool = True) -> List[MeshObject]:
    """ Samples objects poses on a surface.

    The objects are positioned slightly above the surface due to the non-axis aligned nature of used bounding boxes
    and possible non-alignment of the sampling surface (i.e. on the X-Y hyperplane, can be somewhat mitigated with
    precise "up_direction" value), which leads to the objects hovering slightly above the surface. So it is
    recommended to use the PhysicsPositioning module afterwards for realistically looking placements of objects on
    the sampling surface. If placing fails due to collisions, the object will be moved back to the intial pose
    and hidden from rendering.

    :param objects_to_sample: A list of objects that should be sampled above the surface.
    :param surface: Object to place objects_to_sample on.
    :param sample_pose_func: The function to use for sampling the pose of a given object.
    :param max_tries: Amount of tries before giving up on an object (deleting it) and moving to the next one.
    :param min_distance: Minimum distance to the closest other object from objects_to_sample. Center to center.
    :param max_distance: Maximum distance to the closest other object from objects_to_sample. Center to center.
    :param up_direction: Normal vector of the side of surface the objects should be placed on.
    :param check_all_bb_corners_over_surface: If this is True all bounding box corners have to be above the surface,
                                              else only the center of the object has to be above the surface
    :return: The list of placed objects.
    """
    if len(set(objects_to_sample)) != len(objects_to_sample):
        raise ValueError("The given list of objects to sample contains duplicates. "
                         "This leads to complications in the sampling process.")

    if up_direction is None:
        up_direction = np.array([0., 0., 1.])
    else:
        up_direction /= np.linalg.norm(up_direction)

    surface_bounds = surface.get_bound_box()
    surface_height = max(up_direction.dot(corner) for corner in surface_bounds)

    # cache to fasten collision detection
    bvh_cache: Dict[str, mathutils.bvhtree.BVHTree] = {}

    placed_objects: List[MeshObject] = []
    for obj in objects_to_sample:
        print(f"Trying to put {obj.get_name()}")
        initial_pose = obj.get_local2world_mat()
        placed_successfully = False

        for i in range(max_tries):
            sample_pose_func(obj)
            # Remove bvh cache, as object has changed
            if obj.get_name() in bvh_cache:
                del bvh_cache[obj.get_name()]

            if not CollisionUtility.check_intersections(obj, bvh_cache, placed_objects, []):
                print("Collision detected, retrying!")
                continue

            if not _OnSurfaceSampler.check_above_surface(obj, surface, up_direction, check_all_bb_corners_over_surface):
                print("Not above surface, retrying!")
                continue

            _OnSurfaceSampler.drop(obj, up_direction, surface_height)
            # Remove bvh cache, as object has changed
            if obj.get_name() in bvh_cache:
                del bvh_cache[obj.get_name()]

            if not _OnSurfaceSampler.check_above_surface(obj, surface, up_direction, check_all_bb_corners_over_surface):
                print("Not above surface after drop, retrying!")
                continue

            if not _OnSurfaceSampler.check_spacing(obj, placed_objects, min_distance, max_distance):
                print("Bad spacing after drop, retrying!")
                continue

            if not CollisionUtility.check_intersections(obj, bvh_cache, placed_objects, []):
                print("Collision detected after drop, retrying!")
                continue

            print(f"Placed object \"{obj.get_name()}\" successfully at {obj.get_location()} after {i + 1} iterations!")
            placed_objects.append(obj)

            placed_successfully = True
            break

        if not placed_successfully:
            print(f"Giving up on {obj.get_name()}, hiding...")
            obj.hide(True)
            obj.set_local2world_mat(initial_pose)

    return placed_objects


class _OnSurfaceSampler:

    @staticmethod
    def check_above_surface(obj: MeshObject, surface: MeshObject, up_direction: np.ndarray,
                            check_all_bb_corners_over_surface: bool = True) -> bool:
        """ Check if all corners of the bounding box are "above" the surface

        :param obj: Object for which the check is carried out. Type: blender object.
        :param surface: The surface object.
        :param up_direction: The direction that indicates "above" direction.
        :param check_all_bb_corners_over_surface: If this is True all bounding box corners have to be above the surface,
                                                  else only the center of the object has to be above the surface
        :return: True if the bounding box is above the surface, False - if not.
        """
        if check_all_bb_corners_over_surface:
            for point in obj.get_bound_box():
                if not surface.position_is_above_object(point + up_direction, -up_direction,
                                                        check_no_objects_in_between=False):
                    return False
            return True
        center = np.mean(obj.get_bound_box(), axis=0)
        return surface.position_is_above_object(center + up_direction, -up_direction, check_no_objects_in_between=False)

    @staticmethod
    def check_spacing(obj: MeshObject, placed_objects: List[MeshObject], min_distance: float, max_distance: float) \
            -> bool:
        """ Check if object is not too close or too far from previous objects.

        :param obj: Object for which the check is carried out.
        :param placed_objects: A list of already placed objects that should be used for checking spacing.
        :param min_distance: Minimum distance to the closest other object from placed_objects. Center to center.
        :param max_distance: Maximum distance to the closest other object from placed_objects. Center to center.
        :return: True, if the spacing is correct
        """
        closest_distance = None

        for already_placed in placed_objects:
            distance = np.linalg.norm(already_placed.get_location() - obj.get_location())
            if closest_distance is None or distance < closest_distance:
                closest_distance = distance

        return closest_distance is None or (min_distance <= closest_distance <= max_distance)

    @staticmethod
    def drop(obj: MeshObject, up_direction: np.ndarray, surface_height: float):
        """ Moves object "down" until its bounding box touches the bounding box of the surface. This uses bounding boxes
            which are not aligned optimally, this will cause objects to be placed slightly to high.

        :param obj: Object to move. Type: blender object.
        :param up_direction: Vector which points into the opposite drop direction.
        :param surface_height: Height of the surface above its origin.
        """
        obj_bounds = obj.get_bound_box()
        obj_height = min(up_direction.dot(corner) for corner in obj_bounds)

        obj.set_location(obj.get_location() - up_direction * (obj_height - surface_height))
