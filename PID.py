""" 
A simple PID controller class.  

This is a mostly literal C++ -> Python translation of the ROS
control_toolbox Pid class: http://ros.org/wiki/control_toolbox.
"""

#*******************************************************************
# Translated from pid.cpp by Karan Chawla
# December 2017
# See below for original license information:
#*******************************************************************

#******************************************************************* 
# Software License Agreement (BSD License)
#
#  Copyright (c) 2008, Willow Garage, Inc.
#  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions
#  are met:
#
#   * Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#   * Redistributions in binary form must reproduce the above
#     copyright notice, this list of conditions and the following
#     disclaimer in the documentation and/or other materials provided
#     with the distribution.
#   * Neither the name of the Willow Garage nor the names of its
#     contributors may be used to endorse or promote products derived
#     from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
#  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
#  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
#  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
#  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
#  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
#  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.
#******************************************************************* 

import time 
import math 

class PID(object):
    """
	This class implements a generic structure that can be used to
    create a wide range of pid controllers. It can function
    independently or be subclassed to provide more specific controls
    based on a particular control loop.

    In particular, this class implements the standard pid equation:

    $command = -p_{term} - i_{term} - d_{term} $

    where:

    $ p_{term} = p_{gain} * p_{error} $
    $ i_{term} = i_{gain} * i_{error} $
    $ d_{term} = d_{gain} * d_{error} $
    $ i_{error} = i_{error} + p_{error} * dt $
    $ d_{error} = (p_{error} - p_{error last}) / dt $

    given:

    $ p_{error} = p_{state} - p_{target} $.
    """ 
    def __init__(self, pGain, iGain, dGain, iMax, iMin):
        """Constructor, zeros out error values when created and
        initialize Pid-gains and integral term limits.

        Parameters:
          pGain      The proportional gain.
          iGain      The integral gain.
          dGain      The derivative gain.
          iMax       The integral upper limit.
          iMin       The integral lower limit. 
        """
        self.set_gains(pGain, iGain, dGain, iMax, iMin)
        self.reset()

    def reset(self):
    	"""
    	Reset the state of this PID controller
    	"""
        self._pErrorLast = 0.0 # LAst saved postition for derivative gain 
        self._pError = 0.0 # Position error
        self._dError = 0.0 # Derivative error
        self._iError = 0.0 # Integator error
        self._cmd = 0.0 # Command to send
        self._lastTime = None # Used for automatic calculation of dt
    
    def set_gains(self, pGain, iGain, dGain, iMax, iMin): 
        """ Set PID gains for the controller. 

         Parameters:
          p_gain     The proportional gain.
          iGain     The integral gain.
          dGain     The derivative gain.
          iMax      The integral upper limit.
          iMin      The integral lower limit. 
        """ 
        self._pGain = pGain
        self._iGain = iGain
        self._dGain = dGain
        self._iMax = iMax
        self._iMin = iMin

    @property
    def pGain(self):
        """ Read-only access to p_gain. """
        return self._pGain

    @property
    def iGain(self):
        """ Read-only access to iGain. """
        return self._iGain

    @property
    def dGain(self):
        """ Read-only access to dGain. """
        return self._dGain

    @property
    def iMax(self):
        """ Read-only access to iMax. """
        return self._iMax

    @property
    def iMin(self):
        """ Read-only access to iMin. """
        return self._iMin

    @property
    def pError(self):
        """ Read-only access to pError. """
        return self._pError

    @property
    def iError(self):
        """ Read-only access to iError. """
        return self._iError

    @property
    def dError(self):
        """ Read-only access to dError. """
        return self._dError

    @property
    def cmd(self):
        """ Read-only access to the latest command. """
        return self._cmd

    def __str__(self):
        """ String representation of the current state of the controller. """

        result = ""
        result += "p_gain:  " + str(self._pGain) + "\n"
        result += "iGain:  " + str(self._iGain) + "\n"
        result += "dGain:  " + str(self._dGain) + "\n"
        result += "iMax:   " + str(self._iMax) + "\n"
        result += "iMin:   " + str(self._iMin) + "\n"
        result += "pError: " + str(self._pError) + "\n"
        result += "iError: " + str(self._iError) + "\n"
        result += "dError: " + str(self._dError) + "\n"
        result += "cmd:     " + str(self._cmd) + "\n"
        return result

    def update_PID(self, pError, dt=None):
        """  Update the Pid loop with nonuniform time step size.

        Parameters:
          pError  Error since last call (p_state - p_target)
          dt       Change in time since last call, in seconds, or None. 
                   If dt is None, then the system clock will be used to 
                   calculate the time since the last update. 
        """
        if dt == None:
            curTime = time.time()
            if self._lastTime is None:
                self._lastTime = curTime 
            dt = curTime - self._lastTime
            self._lastTime = curTime
            
        self._pError = pError # this is pError = pState-pTarget
        if dt == 0 or math.isnan(dt) or math.isinf(dt):
            return 0.0

        # Calculate proportional contribution to command
        pTerm = self._pGain * self._pError

        # Calculate the integral error
        self._iError += dt * self._pError
        
        # Calculate integral contribution to command
        iTerm = self._iGain * self._iError
        
        # Limit iTerm so that the limit is meaningful in the output
        if iTerm > self._iMax and self._iGain != 0:
            iTerm = self._iMax
            self._iError = iTerm / self._iGain
        elif iTerm < self._iMin and self._iGain != 0:
            iTerm = self._iMin
            self._iError = iTerm / self._iGain
            
        # Calculate the derivative error
        self._dError = (self._pError - self._pErrorLast) / dt
        self._pError_last = self._pError
        
        # Calculate derivative contribution to command 
        dTerm = self._dGain * self._dError
        
        self._cmd = -pTerm - iTerm - dTerm

        return self._cmd

if __name__ == "__main__":
    controller = PID(1.0, 2.0, 3.0, 1.0, -1.0)
    print controller
    controller.update_PID(-1)
    print controller
    controller.update_PID(-.5)
    print controller