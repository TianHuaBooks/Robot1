import numpy as np
import cv2

from supporting_functions import is_close_blacklist

# Identify pixels above the threshold
# Threshold of RGB > 160 does a nice job of identifying ground pixels only
def color_thresh(img, rgb_thresh=(160, 160, 160)):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    above_thresh = (img[:,:,0] > rgb_thresh[0]) \
                & (img[:,:,1] > rgb_thresh[1]) \
                & (img[:,:,2] > rgb_thresh[2])
    # Index the array of zeros with the boolean array and set to 1
    color_select[above_thresh] = 1
    # Return the binary image
    return color_select

# Identify pixels between thresholds
def color_thresh2(img, rgb_thresh_min=(90, 90, 0), rgb_thresh_max=(210, 210, 0)):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be between all three threshold values in RGB
    # threshholds will now contain a boolean array with "True"
    # where threshold was met
    above_thresh = (img[:,:,0] >= rgb_thresh_min[0]) & (img[:,:,0] <= rgb_thresh_max[0]) \
        & (img[:,:,1] >= rgb_thresh_min[1]) & (img[:,:,1] <= rgb_thresh_max[1]) \
        & (img[:,:,2] >= rgb_thresh_min[2]) & (img[:,:,2] <= rgb_thresh_max[2])
    # Index the array of zeros with the boolean array and set to 1
    color_select[above_thresh] = 1
    # Return the binary image
    return color_select

def find_rock(img, rgb_thresh=(100, 100, 50)):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be between all three threshold values in RGB
    # threshholds will now contain a boolean array with "True"
    # where threshold was met
    above_thresh = (img[:,:,0] > rgb_thresh[0])  \
        & (img[:,:,1] > rgb_thresh[1])  \
        & (img[:,:,2] < rgb_thresh[2])
    # Index the array of zeros with the boolean array and set to 1
    color_select[above_thresh] = 1
    # Return the binary image
    return color_select

# Define a function to convert from image coords to rover coords
def rover_coords(binary_img):
    # Identify nonzero pixels
    ypos, xpos = binary_img.nonzero()
    # Calculate pixel positions with reference to the rover position being at the 
    # center bottom of the image.  
    x_pixel = -(ypos - binary_img.shape[0]).astype(np.float)
    y_pixel = -(xpos - binary_img.shape[1]/2 ).astype(np.float)
    return x_pixel, y_pixel


# Define a function to convert to radial coords in rover space
def to_polar_coords(x_pixel, y_pixel):
    # Convert (x_pixel, y_pixel) to (distance, angle) 
    # in polar coordinates in rover space
    # Calculate distance to each pixel
    dist = np.sqrt(x_pixel**2 + y_pixel**2)
    # Calculate angle away from vertical for each pixel
    angles = np.arctan2(y_pixel, x_pixel)
    return dist, angles

# Define a function to map rover space pixels to world space
def rotate_pix(xpix, ypix, yaw):
    # Convert yaw to radians
    yaw_rad = yaw * np.pi / 180
    xpix_rotated = (xpix * np.cos(yaw_rad)) - (ypix * np.sin(yaw_rad))
                            
    ypix_rotated = (xpix * np.sin(yaw_rad)) + (ypix * np.cos(yaw_rad))
    # Return the result  
    return xpix_rotated, ypix_rotated

def translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale): 
    # Apply a scaling and a translation
    xpix_translated = (xpix_rot / scale) + xpos
    ypix_translated = (ypix_rot / scale) + ypos
    # Return the result  
    return xpix_translated, ypix_translated


# Define a function to apply rotation and translation (and clipping)
# Once you define the two functions above this function should work
def pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale):
    # Apply rotation
    xpix_rot, ypix_rot = rotate_pix(xpix, ypix, yaw)
    # Apply translation
    xpix_tran, ypix_tran = translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale)
    # Perform rotation, translation and clipping all at once
    x_pix_world = np.clip(np.int_(xpix_tran), 0, world_size - 1)
    y_pix_world = np.clip(np.int_(ypix_tran), 0, world_size - 1)
    # Return the result
    return x_pix_world, y_pix_world

# utility to check whether a rock is reachable
def is_rock_reachable(rock_angle, nav_angles):
    result = False
    if nav_angles is not None and len(nav_angles) > 1:
        ang_min = min(nav_angles)
        ang_max = max(nav_angles)
        result = (rock_angle >= ang_min) and (rock_angle <= ang_max)
    
    if result == False:
        print("rock_angle:", rock_angle, " nav_angles:", nav_angles)
    return result

# Define a function to perform a perspective transform
def perspect_transform(img, src, dst):
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))# keep same size as input image
    mask = cv2.warpPerspective(np.ones_like(img[:,:,0]), M, (img.shape[1], img.shape[0]))
    return warped, mask

dbg_count = 0
# Apply the above functions in succession and update the Rover state accordingly
def perception_step(Rover):
    # Perform perception steps to update Rover()
    # TODO: 
    # NOTE: camera image is coming to you in Rover.img
    image = Rover.img.copy()
    # 1) Define source and destination points for perspective transform
    # Define calibration box in source (actual) and destination (desired) coordinates
    # These source and destination points are defined to warp the image
    # to a grid where each 10x10 pixel square represents 1 square meter
    # The destination box will be 2*dst_size on each side
    dst_size = 5
    # Set a bottom offset to account for the fact that the bottom of the image
    # is not the position of the rover but a bit in front of it
    # this is just a rough guess, feel free to change it!
    bottom_offset = 6
    source = np.float32([[14, 140], [301 ,140],[200, 96], [118, 96]])
    destination = np.float32([[image.shape[1]/2 - dst_size, image.shape[0] - bottom_offset], [image.shape[1]/2 + dst_size, image.shape[0] - bottom_offset], [image.shape[1]/2 + dst_size, image.shape[0] - 2*dst_size - bottom_offset], [image.shape[1]/2 - dst_size, image.shape[0] - 2*dst_size - bottom_offset]])
    
    # 2) Apply perspective transform
    warped, mask = perspect_transform(image, source, destination)
    # 3) Apply color threshold to identify navigable terrain/obstacles/rock samples
    terrain = color_thresh(warped)
    rocks = find_rock(warped)
    obstacle = np.absolute(np.float32(terrain) - 1) * mask
    
    global dbg_count
    if dbg_count == 0:
        dbg_count += 1
        dbg_img = image.copy()
        cv2.polylines(dbg_img, np.int32([source]), True, (0, 0, 255), 1)
        cv2.imwrite("dbg_img.jpg", dbg_img)
        cv2.polylines(warped, np.int32([destination]), True, (0, 0, 255), 3)
        cv2.imwrite("warped.jpg", warped)
        cv2.imwrite("obstacle.jpg", obstacle)
        cv2.imwrite("rocks.jpg", rocks)

    Rover.vision_image[:,:,0] = obstacle * 255
    Rover.vision_image[:,:,2] = terrain * 255
    # 4) Update Rover.vision_image (this will be displayed on left side of screen)
        # Example: Rover.vision_image[:,:,0] = obstacle color-thresholded binary image
        #          Rover.vision_image[:,:,1] = rock_sample color-thresholded binary image
        #          Rover.vision_image[:,:,2] = navigable terrain color-thresholded binary image

    # 5) Convert map image pixel values to rover-centric coords
    xpix, ypix = rover_coords(terrain)
    xpix_obstacle, ypix_obstacle = rover_coords(obstacle)
    
    # 6) Convert rover-centric pixel values to world coordinates
    scale = 2*dst_size
    world_size = Rover.worldmap.shape[0]
    obstacle_x_world, obstacle_y_world = pix_to_world(xpix_obstacle, ypix_obstacle, Rover.pos[0], Rover.pos[1], Rover.yaw, world_size, scale)
    navigable_x_world, navigable_y_world = pix_to_world(xpix, ypix, Rover.pos[0], Rover.pos[1], Rover.yaw, world_size, scale)

    # 7) Update Rover worldmap (to be displayed on right side of screen)
        # Example: Rover.worldmap[obstacle_y_world, obstacle_x_world, 0] += 1
        #          Rover.worldmap[rock_y_world, rock_x_world, 1] += 1
        #          Rover.worldmap[navigable_y_world, navigable_x_world, 2] += 1
    Rover.worldmap[obstacle_y_world, obstacle_x_world, 0] += 1
    Rover.worldmap[navigable_y_world, navigable_x_world, 2] += 1

        # 8) Convert rover-centric pixel positions to polar coordinates
    dist, angle = to_polar_coords(xpix, ypix)

        # check rocks
    if rocks.any():
        xpix_rocks, ypix_rocks = rover_coords(rocks)
        rock_x_world, rock_y_world = pix_to_world(xpix_rocks, ypix_rocks, Rover.pos[0], Rover.pos[1], Rover.yaw, world_size, scale)
        rock_dist, rock_ang = to_polar_coords(xpix_rocks, ypix_rocks)
        rock_idx = 0
        if len(rock_dist) > 1:
            rock_idx = np.argmin(rock_dist)
        rock_xcen = rock_x_world[rock_idx]
        rock_ycen = rock_y_world[rock_idx]
        # try to avoid trap (black list)
        if is_close_blacklist(rock_xcen, rock_ycen):
            print("HIT blacklist rock_xcen:", rock_xcen, " rock_ycen:", rock_ycen)
            if len(rock_dist) == 1:
                Rover.rock_picked_pos[0] = rock_xcen
                Rover.rock_picked_pos[1] = rock_ycen
            else:
                rock_idx -= 1
                rock_xcen = rock_x_world[rock_idx]
                rock_ycen = rock_y_world[rock_idx]
        
        Rover.worldmap[rock_ycen, rock_xcen, 1] = 255
        Rover.vision_image[:,:,1] = rocks * 255
        # go after rock
        Rover.nearest_rock_angle = None
            #if np.fabs(rock_ang[rock_idx]) > 0.2 and \
            #np.fabs(rock_dist[rock_idx]) > 6.5 and \
            #np.fabs(Rover.rock_picked_pos[0] - rock_xcen) > 8.0 and \
            #np.fabs(Rover.rock_picked_pos[1] - rock_ycen) > 8.0:
        if np.fabs(Rover.rock_picked_pos[0] - rock_xcen) > 8.0 and \
            np.fabs(Rover.rock_picked_pos[1] - rock_ycen) > 8.0:
            Rover.nearest_rock_angle = rock_ang[rock_idx]
            print("Go after rock rock dist:", rock_dist[rock_idx], " angle:", rock_ang[rock_idx], " rock_ycen:", rock_ycen, " rock_xcen:", rock_xcen)
    else:
        Rover.vision_image[:,:,1] = 0

    # Update Rover pixel distances and angles
    # Rover.nav_dists = rover_centric_pixel_distances
    # Rover.nav_angles = rover_centric_angles
    Rover.nav_dists = dist
    Rover.nav_angles = angle
    
    return Rover
