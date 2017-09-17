import numpy as np
from supporting_functions import is_close_blacklist

def get_steer_angle(Rover, angles):
    if len(angles) > 1:
        val = np.mean(angles)
        if Rover.map_percent1 == Rover.map_percent2:
            if Rover.total_time > 800:
                idx = np.random.randint(0, len(angles))
                val = angles[idx]
            else:
                sigma = np.std(angles) * Rover.total_time / 1000.0
                print("map_percent mean:", val, " sigma:", sigma)
                if Rover.total_time > 500:
                    val += sigma
                else:
                    val -= sigma
        return np.clip(val * 180/np.pi, -15, 15)
    else:
        print("get_steer_angle get len(angles):", len(angles))
        return 0

# This is where you can build a decision tree for determining throttle, brake and steer 
# commands based on the output of the perception_step() function
def decision_step(Rover):

    # Implement conditionals to decide what to do given perception data
    # Here you're all set up with some basic functionality but you'll need to
    # improve on this decision tree to do a good job of navigating autonomously!

    # Example:
    # Check if we have vision data to make decisions with
    if Rover.nav_angles is not None:
        # Check for Rover.mode status
        if Rover.mode == 'forward': 
            # Check the extent of navigable terrain
            if Rover.nearest_rock_angle is not None and not Rover.send_pickup and not Rover.picking_up:
                Rover.brake = 0
                Rover.steer = np.clip(Rover.nearest_rock_angle * 180/np.pi, -15, 15)
        
            elif len(Rover.nav_angles) >= Rover.stop_forward:
                # If mode is forward, navigable terrain looks good 
                # and velocity is below max, then throttle 
                if Rover.vel < Rover.max_vel:
                    # Set throttle value to throttle setting
                    Rover.throttle = Rover.throttle_set
                else: # Else coast
                    Rover.throttle = 0
                Rover.brake = 0
                # Set steering to average angle clipped to the range +/- 15
                if Rover.vel < 0.0:
                    Rover.throttle = Rover.throttle_set
                    Rover.steer = get_steer_angle(Rover, Rover.nav_angles)
                    print("Try to get out steer:", Rover.steer, " throttle:", Rover.throttle)
                else:
                    Rover.steer = get_steer_angle(Rover, Rover.nav_angles)
            # If there's a lack of navigable terrain pixels then go to 'stop' mode
            elif len(Rover.nav_angles) < Rover.stop_forward:
                    # Set mode to "stop" and hit the brakes!
                    Rover.throttle = 0
                    # Set brake to stored brake value
                    Rover.brake = Rover.brake_set
                    Rover.steer = 0
                    Rover.mode = 'stop'

        # If we're already in "stop" mode then make different decisions
        elif Rover.mode == 'stop':
            # If we're in stop mode but still moving keep braking
            if Rover.vel > 0.2:
                Rover.throttle = 0
                Rover.brake = Rover.brake_set
                Rover.steer = 0
            # If we're not moving (vel < 0.2) then do something else
            elif Rover.vel <= 0.2:
                # Now we're stopped and we have vision data to see if there's a path forward
                if len(Rover.nav_angles) < Rover.go_forward:
                    Rover.throttle = 0
                    # Release the brake to allow turning
                    Rover.brake = 0
                    # Turn range is +/- 15 degrees, when stopped the next line will induce 4-wheel turning
                    Rover.steer = -15 # Could be more clever here about which way to turn
                # If we're stopped but see sufficient navigable terrain in front then go!
                if len(Rover.nav_angles) >= Rover.go_forward:
                    # Set throttle back to stored value
                    Rover.throttle = Rover.throttle_set
                    # Release the brake
                    Rover.brake = 0
                    # Set steer to mean angle
                    Rover.steer = get_steer_angle(Rover, Rover.nav_angles)
                    Rover.mode = 'forward'
        elif Rover.mode == 'stuck':
            if  np.fabs(Rover.vel) > 0.4:
                Rover.throttle = Rover.throttle_set
                Rover.brake = 0
                Rover.steer = get_steer_angle(Rover, Rover.nav_angles)
                Rover.mode = 'forward'
                Rover.stuck_time = None
                Rover.retry_count = 0
                print("forward again vel:", Rover.vel, " throttle:", Rover.throttle)
            else:
                # try to back off first, then with states to retry
                Rover.throttle = Rover.throttle_set * -1.1
                if (Rover.total_time - Rover.stuck_time) > 12.5:
                    Rover.retry_count += 1
                    cycles = Rover.retry_count / 350
                    if (int(cycles) % 2):
                        Rover.throttle = Rover.throttle * -1.1
                        print(" Flip throttle:", Rover.throttle, " count:", Rover.retry_count)
                    count = Rover.retry_count % 350
                    if count > 300:
                        Rover.steer = np.clip(np.mean(Rover.nav_angles) * 180/np.pi, -15, 15)
                    elif count > 250:
                        Rover.steer = 15
                    elif count > 200:
                        Rover.steer = 7.5
                    elif count > 150:
                        Rover.steer = 0.0
                    elif count > 100:
                        Rover.steer = -7.5
                    else:
                        Rover.steer = -15
                else:
                    Rover.steer = 0
                print("back off vel:", Rover.vel, " throttle:", Rover.throttle, " count:", Rover.retry_count, " steer:", Rover.steer, " total_time", Rover.total_time," stuck_time", Rover.stuck_time)
    # Just to make the rover do something
    # even if no modifications have been made to the code
    else:
        Rover.throttle = Rover.throttle_set
        Rover.steer = 0
        Rover.brake = 0

    # check whether the Robo is stuck
    if np.fabs(Rover.vel) < 0.1 and Rover.mode != 'stuck':
        if Rover.stuck_time is None:
            Rover.stuck_time = Rover.total_time
            Rover.stuck_pos = Rover.pos
        elif (Rover.total_time - Rover.stuck_time) > 10:
            print("stuck_pos:", Rover.stuck_pos, " pos:", Rover.pos, " total_time", Rover.total_time," stuck_time", Rover.stuck_time)
            Rover.mode = 'stuck'
            Rover.retry_count = 0
            Rover.brake = 0
            Rover.throttle = -Rover.throttle_set

    if Rover.vel > 0.5:
        Rover.stuck_time = None

    # If in a state where want to pickup a rock send pickup command
    if Rover.near_sample:
        if Rover.vel <= 0.75 and not Rover.picking_up:
            Rover.send_pickup = True
            # Set throttle back to stored value
            Rover.throttle = Rover.throttle_set
            # Release the brake
            Rover.brake = 0
            # Set steer to mean angle
            Rover.steer = np.clip(np.mean(Rover.nav_angles * 180/np.pi), -15, 15)
            Rover.mode = 'forward'
            Rover.nearest_rock_angle = None
            #Rover.rock_collected.append([Rover.pos[0], Rover.pos[1]])
            Rover.rock_picked_pos = Rover.pos
            print("pick up @:", Rover.pos)
        else:
            # Set mode to "stop" and hit the brakes!
            Rover.throttle = 0
            # Set brake to stored brake value
            Rover.brake = Rover.brake_set
            Rover.steer = 0
            Rover.mode = 'stop'
            #print("Rover.near_sample Rover.vel =", Rover.vel)
    
    return Rover

