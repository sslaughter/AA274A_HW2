import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from utils import plot_line_segments

class AStar(object):
    """Represents a motion planning problem to be solved using A*"""

    def __init__(self, statespace_lo, statespace_hi, x_init, x_goal, occupancy, resolution=1):
        self.statespace_lo = statespace_lo         # state space lower bound (e.g., [-5, -5])
        self.statespace_hi = statespace_hi         # state space upper bound (e.g., [5, 5])
        self.occupancy = occupancy                 # occupancy grid (a DetOccupancyGrid2D object)
        self.resolution = resolution               # resolution of the discretization of state space (cell/m)
        self.x_init = self.snap_to_grid(x_init)    # initial state
        self.x_goal = self.snap_to_grid(x_goal)    # goal state

        self.closed_set = set()    # the set containing the states that have been visited
        self.open_set = set()      # the set containing the states that are condidate for future expension

        self.est_cost_through = {}  # dictionary of the estimated cost from start to goal passing through state (often called f score)
        self.cost_to_arrive = {}    # dictionary of the cost-to-arrive at state from start (often called g score), indexed by states
        self.came_from = {}         # dictionary keeping track of each state's parent to reconstruct the path

        self.open_set.add(self.x_init)
        self.cost_to_arrive[self.x_init] = 0
        self.est_cost_through[self.x_init] = self.distance(self.x_init,self.x_goal)

        self.path = None        # the final path as a list of states

    def is_free(self, x):
        """
        Checks if a give state x is free, meaning it is inside the bounds of the map and
        is not inside any obstacle.
        Inputs:
            x: state tuple
        Output:
            Boolean True/False
        Hint: self.occupancy is a DetOccupancyGrid2D object, take a look at its methods for what might be
              useful here
        """
        ########## Code starts here ##########



        # Check if the state is within the bounds of the map (adapted from is_free() function within class defined below)
        for dim in range(len(x)):
            if x[dim] < self.statespace_lo[dim] or x[dim] > self.statespace_hi[dim]: 
                return False # if state is not within bounds of map, immediately return False
            else:
                pass

        if self.occupancy.is_free(x): # Check if the state is within the obstacles using defined is_free function from occupancy object
            return True # State is not in obstacle, but is in map, so return True
           
        else:
            return False # State is within the map, but also within an obstacle, so return False

        
        ########## Code ends here ##########

    def distance(self, x1, x2):
        """
        Computes the Euclidean distance between two states.
        Inputs:
            x1: First state tuple
            x2: Second state tuple
        Output:
            Float Euclidean distance

        HINT: This should take one line. Tuples can be converted to numpy arrays using np.array().
        """
        ########## Code starts here ##########
        dist = np.sqrt((x1[0]-x2[0])**2 + (x1[1]-x2[1])**2)
        return dist
    
        ########## Code ends here ##########

    def snap_to_grid(self, x):
        """ Returns the closest point on a discrete state grid
        Input:
            x: tuple state
        Output:
            A tuple that represents the closest point to x on the discrete state grid
        """
        return (self.resolution*round(x[0]/self.resolution), self.resolution*round(x[1]/self.resolution))

    def get_neighbors(self, x):
        """
        Gets the FREE neighbor states of a given state x. Assumes a motion model
        where we can move up, down, left, right, or along the diagonals by an
        amount equal to self.resolution.
        Input:
            x: tuple state
        Ouput:
            List of neighbors that are free, as a list of TUPLES

        HINTS: Use self.is_free to check whether a given state is indeed free.
               Use self.snap_to_grid (see above) to ensure that the neighbors
               you compute are actually on the discrete grid, i.e., if you were
               to compute neighbors by adding/subtracting self.resolution from x, 
               numerical errors could creep in over the course of many additions 
               and cause grid point equality checks to fail. To remedy this, you 
               should make sure that every neighbor is snapped to the grid as it 
               is computed.
        """
        neighbors = []
        ########## Code starts here ##########
        step = int(self.resolution)
        pos_neighbors = [(x[0],x[1]+step),
                        (x[0],x[1]-step),
                        (x[0]-step,x[1]+step),
                        (x[0]-step,x[1]-step),
                        (x[0]-step,x[1]),
                        (x[0]+step,x[1]+step),
                        (x[0]+step,x[1]-step),
                        (x[0]+step,x[1])]
        
        for neighbor in pos_neighbors:
            self.snap_to_grid(neighbor)
            if self.is_free(neighbor):
                neighbors.append(neighbor)
            else:
                pass

        
        ########## Code ends here ##########
        return neighbors

    def find_best_est_cost_through(self):
        """
        Gets the state in open_set that has the lowest est_cost_through
        Output: A tuple, the state found in open_set that has the lowest est_cost_through
        """
        return min(self.open_set, key=lambda x: self.est_cost_through[x])

    def reconstruct_path(self):
        """
        Use the came_from map to reconstruct a path from the initial location to
        the goal location
        Output:
            A list of tuples, which is a list of the states that go from start to goal
        """
        path = [self.x_goal]
        current = path[-1]
        while current != self.x_init:
            path.append(self.came_from[current])
            current = path[-1]
        path = list(reversed(path))

        #for r in range(len(path)):
          #  path[r] = np.asarray(path[r]) # convert the list objects to arrays
        #return list(reversed(path))
        #print(type(path[0]))
        return path

    def plot_path(self, fig_num=0, show_init_label=True):
        """Plots the path found in self.path and the obstacles"""
        if not self.path:
            return

        self.occupancy.plot(fig_num)

        solution_path = np.array(self.path) * self.resolution
        plt.plot(solution_path[:,0],solution_path[:,1], color="green", linewidth=2, label="A* solution path", zorder=10)
        plt.scatter([self.x_init[0]*self.resolution, self.x_goal[0]*self.resolution], [self.x_init[1]*self.resolution, self.x_goal[1]*self.resolution], color="green", s=30, zorder=10)
        if show_init_label:
            plt.annotate(r"$x_{init}$", np.array(self.x_init)*self.resolution + np.array([.2, .2]), fontsize=16)
        plt.annotate(r"$x_{goal}$", np.array(self.x_goal)*self.resolution + np.array([.2, .2]), fontsize=16)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.03), fancybox=True, ncol=3)

        plt.axis([0, self.occupancy.width, 0, self.occupancy.height])

    def plot_tree(self, point_size=15):
        plot_line_segments([(x, self.came_from[x]) for x in self.open_set if x != self.x_init], linewidth=1, color="blue", alpha=0.2)
        plot_line_segments([(x, self.came_from[x]) for x in self.closed_set if x != self.x_init], linewidth=1, color="blue", alpha=0.2)
        px = [x[0] for x in self.open_set | self.closed_set if x != self.x_init and x != self.x_goal]
        py = [x[1] for x in self.open_set | self.closed_set if x != self.x_init and x != self.x_goal]
        plt.scatter(px, py, color="blue", s=point_size, zorder=10, alpha=0.2)

    def solve(self):
        """
        Solves the planning problem using the A* search algorithm. It places
        the solution as a list of tuples (each representing a state) that go
        from self.x_init to self.x_goal inside the variable self.path
        Input:
            None
        Output:
            Boolean, True if a solution from x_init to x_goal was found

        HINTS:  We're representing the open and closed sets using python's built-in
                set() class. This allows easily adding and removing items using
                .add(item) and .remove(item) respectively, as well as checking for
                set membership efficiently using the syntax "if item in set".
        """
        ########## Code starts here ##########
        # Based on pseudocode in hw handout:
        # Open/closed sets have already been initialized by the class, as have the initial cost values
        found_goal = False

        while len(self.open_set) > 0:
            x_current = self.find_best_est_cost_through() # Get the state within the open set that has the lowest cost to arrive, uses defined dictionary keyed by states
            if x_current == self.x_goal:
                found_goal = True
                break
            else:
                pass
            self.open_set.remove(x_current)
            self.closed_set.add(x_current)
            for p_neigh in self.get_neighbors(x_current):
                if p_neigh in self.closed_set: # Check if the neighbor is in the closed set (has already been looked at), if it has skip
                    continue
                ten_cta = self.cost_to_arrive[x_current] + self.distance(x_current,p_neigh) # get cost to arrive of current neighbor being evaluated; this is the cost to get there through x_current
                if p_neigh not in self.open_set: # add new neighbor into open set if it's not already there
                    self.open_set.add(p_neigh)
                elif ten_cta > self.cost_to_arrive[p_neigh]: # if the neighbor is already in open set, check if the cost to get there along the current path is less than the cost to get there when it was originally added. If it's not, skip this neighbor.
                    continue
                self.came_from[p_neigh] = x_current
                self.cost_to_arrive[p_neigh] = ten_cta
                self.est_cost_through[p_neigh] = ten_cta + self.distance(p_neigh,self.x_goal)


        if found_goal == True:
            self.path = self.reconstruct_path()
            return True
        else:
            return False

        
        ########## Code ends here ##########

class DetOccupancyGrid2D(object):
    """
    A 2D state space grid with a set of rectangular obstacles. The grid is
    fully deterministic
    """
    def __init__(self, width, height, obstacles):
        self.width = width
        self.height = height
        self.obstacles = obstacles

    def is_free(self, x):
        """Verifies that point is not inside any obstacles"""
        for obs in self.obstacles:
            inside = True
            for dim in range(len(x)):
                if x[dim] < obs[0][dim] or x[dim] > obs[1][dim]:
                    inside = False
                    break
            if inside:
                return False
        return True

    def plot(self, fig_num=0):
        """Plots the space and its obstacles"""
        fig = plt.figure(fig_num)
        ax = fig.add_subplot(111, aspect='equal')
        for obs in self.obstacles:
            ax.add_patch(
            patches.Rectangle(
            obs[0],
            obs[1][0]-obs[0][0],
            obs[1][1]-obs[0][1],))
        ax.set(xlim=(0,self.width), ylim=(0,self.height))
