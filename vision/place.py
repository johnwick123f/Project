# took me literally the most time lol, simple function for estimation a place point. 
import numpy as np

def place_estimation(object_mask, surface_mask, proximity):
    # Convert object and surface masks to grayscale
    object_gray = cv2.cvtColor(object_mask, cv2.COLOR_BGR2GRAY)
    surface_gray = cv2.cvtColor(surface_mask, cv2.COLOR_BGR2GRAY)

    # Find indices of object and surface pixels
    object_indices = np.argwhere(object_gray > 0)
    surface_indices = np.argwhere(surface_gray > 0)

    # Calculate centroid of object
    object_centroid = np.mean(object_indices, axis=0)

    # Calculate distances between object centroid and surface points
    distances = np.linalg.norm(surface_indices - object_centroid, axis=1)

    # Calculate the proximity threshold based on the specified value
    proximity_threshold = np.percentile(distances, proximity)

    # Filter surface points based on proximity
    valid_indices = np.where(distances >= proximity_threshold)

    # Filter the surface points
    valid_surface_points = surface_indices[valid_indices]

    # Find the closest point on the surface
    closest_point_index = np.argmin(np.linalg.norm(valid_surface_points - object_centroid, axis=1))
    closest_point = valid_surface_points[closest_point_index]

    return closest_point
