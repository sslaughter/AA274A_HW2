import numpy as np
from utils import wrapToPi

# command zero velocities once we are this close to the goal
RHO_THRES = 0.15
ALPHA_THRES = 0.1
DELTA_THRES = 0.1

class PoseController:
    """ Pose stabilization controller """
    def __init__(self, k1, k2, k3, V_max=0.5, om_max=1):
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3

        self.V_max = V_max
        self.om_max = om_max

    def load_goal(self, x_g, y_g, th_g):
        """ Loads in a new goal position """
        self.x_g = x_g
        self.y_g = y_g
        self.th_g = th_g


    def compute_control(self, x, y, th, t):
        """
        Inputs:
            x,y,th: Current state
            t: Current time (you shouldn't need to use this)
        Outputs: 
            V, om: Control actions

        Hints: You'll need to use the wrapToPi function. The np.sinc function
        may also be useful, look up its documentation
        """
        ########## Code starts here ##########
        x = x-self.x_g
        y = y-self.y_g

        rho = np.sqrt(x**2 + y**2)
        alph = wrapToPi(np.arctan2(y,x)- th + np.pi)
        delt = wrapToPi(alph + th-self.th_g)
        #delt = wrapToPi(alph + th-self.th_g) # incorporate non-zero theta_goal by subtracting th_g from delta calculation

        if (rho <= RHO_THRES) and (np.absolute(alph) <= ALPHA_THRES) and (np.absolute(delt) <= DELTA_THRES):
            V=0
            om = 0
        else:
            if rho > RHO_THRES: #If the distance threshold is not met, do normal control
                V = self.k1*rho*np.cos(alph)
                om = self.k2*alph + self.k1*(np.sinc(alph/np.pi)*np.cos(alph))*(alph+self.k3*delt) # use sinc as replacement of sin(alph)/alph since that will diverge at small alpha
            else: # If all that's needed is rotation, alpha becomes too small to generate significant control input, so modify om control to simple proportional based on theta error.
                V = 0
                om = self.k2*wrapToPi(self.th_g-th)
            

            ########## Code ends here ##########

            # apply control limits
        V = np.clip(V, -self.V_max, self.V_max)
        om = np.clip(om, -self.om_max, self.om_max)

        return V, om
