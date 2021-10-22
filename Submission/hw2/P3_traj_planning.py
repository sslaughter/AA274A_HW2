import numpy as np
from P1_astar import DetOccupancyGrid2D, AStar
from P2_rrt import *
import scipy.interpolate
import matplotlib.pyplot as plt
from HW1.P1_differential_flatness import *
#from HW1.P1_differential_flatness import compute_controls
from HW1.P2_pose_stabilization import *
from HW1.P3_trajectory_tracking import *

class SwitchingController(object):
    """
    Uses one controller to initially track a trajectory, then switches to a 
    second controller to regulate to the final goal.
    """
    def __init__(self, traj_controller, pose_controller, t_before_switch):
        self.traj_controller = traj_controller
        self.pose_controller = pose_controller
        self.t_before_switch = t_before_switch # Switch occurs at t_final - t_before_switch

    def compute_control(self, x, y, th, t):
        """
        Inputs:
            (x,y,th): Current state 
            t: Current time

        Outputs:
            V, om: Control actions
        """
        # Hint: Both self.traj_controller and self.pose_controller have compute_control() functions. 
        #       When should each be called? Make use of self.t_before_switch and 
        #       self.traj_controller.traj_times.
        ########## Code starts here ##########
        #print(self.traj_controller.traj_times)
        if t <= self.traj_controller.traj_times[-1]-self.t_before_switch:
            #print("I'm in trajectory controller, t = %f" % t)
            return self.traj_controller.compute_control(x,y,th,t)
        else:
            #print("I'm in pose controller, t = %f" % t)
            return self.pose_controller.compute_control(x,y,th,t)

        ########## Code ends here ##########

def compute_smoothed_traj(path, V_des, alpha, dt):
    """
    Fit cubic spline to a path and generate a resulting trajectory for our
    wheeled robot.

    Inputs:
        path (np.array [N,2]): Initial path
        V_des (float): Desired nominal velocity, used as a heuristic to assign nominal
            times to points in the initial path
        alpha (float): Smoothing parameter (see documentation for
            scipy.interpolate.splrep)
        dt (float): Timestep used in final smooth trajectory
    Outputs:
        traj_smoothed (np.array [N,7]): Smoothed trajectory
        t_smoothed (np.array [N]): Associated trajectory times
    Hint: Use splrep and splev from scipy.interpolate
    """
    ########## Code starts here ##########
    # Hint 1 - Determine nominal time for each point in the path using V_des
    # Hint 2 - Use splrep to determine cubic coefficients that best fit given path in x, y
    # Hint 3 - Use splev to determine smoothed paths. The "der" argument may be useful.
    dist = []
    time = []
    time.append(0.0)
    path = np.asarray(path)

    for i in range(len(path)-1):
        dist.append(np.linalg.norm(path[i+1,:] - path[i,:]))
        time.append((dist[i]/V_des)+time[i])

    #path_no_duplicates = np.empty([1,2])
    #ind = 0
    #remove x duplicates
    """
    for x in path[:,0]:
        if x not in path_no_duplicates[:,0]:
            print("add to no dup")
            print(ind)
            path_no_duplicates = np.vstack((path_no_duplicates,path[ind]))
        else:
            pass
        ind+=1
    

    print(path_no_duplicates)
    path_no_duplicates = np.delete(path_no_duplicates,0,0)
    print("duplicates removed")
    print(path_no_duplicates)
    
    """
    #Implementation greatly assisted by Pei-Chen Wu's EdX answer
    #print(time)
    #print(path)

    spline_x = scipy.interpolate.splrep(time,path[:,0],k=3,s=alpha) #working to this point
    spline_y = scipy.interpolate.splrep(time,path[:,1],k=3,s=alpha)

    num_time_points = int(time[-1]/dt)
    #rint(num_time_points)

    t_smoothed = np.linspace(0,time[-1],num_time_points,endpoint=True)

    #print("Number of points in time: %f" % len(t_smoothed))

    x_d = scipy.interpolate.splev(t_smoothed,spline_x,der=0)
    xd_d = scipy.interpolate.splev(t_smoothed,spline_x,der=1)
    xdd_d = scipy.interpolate.splev(t_smoothed,spline_x,der=2)

    y_d = scipy.interpolate.splev(t_smoothed,spline_y,der=0)
    yd_d = scipy.interpolate.splev(t_smoothed,spline_y,der=1)
    ydd_d = scipy.interpolate.splev(t_smoothed,spline_y,der=2)


    theta_d = np.arctan2(yd_d,xd_d)

    #evaluated_s_path = scipy.interpolate.splev(path[:,0],spline_path, der=2)
    #print(evaluated_s_path)


    
    ########## Code ends here ##########
    traj_smoothed = np.stack([x_d, y_d, theta_d, xd_d, yd_d, xdd_d, ydd_d]).transpose()
    return traj_smoothed, t_smoothed

def modify_traj_with_limits(traj, t, V_max, om_max, dt):
    """
    Modifies an existing trajectory to satisfy control limits and
    interpolates for desired timestep.

    Inputs:
        traj (np.array [N,7]): original trajecotry
        t (np.array [N]): original trajectory times
        V_max, om_max (float): control limits
        dt (float): desired timestep
    Outputs:
        t_new (np.array [N_new]) new timepoints spaced dt apart
        V_scaled (np.array [N_new])
        om_scaled (np.array [N_new])
        traj_scaled (np.array [N_new, 7]) new rescaled traj at these timepoints
    Hint: This should almost entirely consist of calling functions from Problem Set 1
    Hint: Take a close look at the code within compute_traj_with_limits() and interpolate_traj() 
          from P1_differential_flatness.py
    """
    ########## Code starts here ##########

    V,om = compute_controls(traj=traj)
    s = compute_arc_length(V, t)
    V_tilde = rescale_V(V, om, V_max, om_max)
    tau = compute_tau(V_tilde, s)
    om_tilde = rescale_om(V, om, V_tilde)
    s_f = State(x=traj[-1,0],y=traj[-1,1],V=V_tilde[-1],th=traj[-1,2]) # With help from Cole Maxwell/Somrita on EdX
    t_new, V_scaled, om_scaled, traj_scaled = interpolate_traj(traj, tau, V_tilde, om_tilde, dt, s_f)

    ########## Code ends here ##########

    return t_new, V_scaled, om_scaled, traj_scaled
