import os
import sys

import numpy as np

import crocoddyl
import example_robot_data
import pinocchio
import time


from pinocchio.robot_wrapper import RobotWrapper
from pinocchio.utils import *
from crocoddyl.utils.quadruped import SimpleQuadrupedalGaitProblem, plotSolution
from quadruped_util import A1andLaikagoGaitProblem

WITHDISPLAY = 'display' in sys.argv or 'CROCODDYL_DISPLAY' in os.environ
WITHPLOT = 'plot' in sys.argv or 'CROCODDYL_PLOT' in os.environ

# Load Laicago Model
URDF_FILENAME = "a1.urdf"
A1modelPath = "/home/naza/Desktop/motion_imitation/"

A1 = RobotWrapper.BuildFromURDF(A1modelPath + URDF_FILENAME, [A1modelPath], pinocchio.JointModelFreeFlyer())

# Add free flyer joint limits
rmodel = A1.model
ub = rmodel.upperPositionLimit
ub[:7] = 1
rmodel.upperPositionLimit = ub
lb = rmodel.lowerPositionLimit
lb[:7] = -1
rmodel.lowerPositionLimit = lb

# Setting up the 3d walking problem FR, FL, RR, RL
# mapping from anymal to Laikago/A1
# lf (left_forward) = FL (forward_left)
# rf (right_forward) = FR (forward_right)
# lh (left_hinder) = RL (rear_left)
# rh (right_hinder) = RR (rear_right)
lfFoot, rfFoot, lhFoot, rhFoot = 'FL_foot', 'FR_foot', 'RL_foot', 'RR_foot'
gait = A1andLaikagoGaitProblem(rmodel, lfFoot, rfFoot, lhFoot, rhFoot)

# Defining the initial state of the robot
q0 = np.array([0., 0., 0.336, 0, 0, 0, 1, 0, 0.6, -1.15, 0, 0.6, -1.15, 0, 0.6, -1.15, 0, 0.6, -1.15])
v0 = pinocchio.utils.zero(rmodel.nv)
x0 = np.concatenate([q0, v0])

# Defining the gait, declare ddp solver
walking_gait = {'stepLength': 0.25, 'stepHeight': 0.2, 'timeStep': 5e-3, 'stepKnots': 40, 'supportKnots': 2}
# trotting_gait = {'stepLength': 0.15, 'stepHeight': 0.1, 'timeStep': 1e-2, 'stepKnots': 25, 'supportKnots': 2}
# jumping_gait = {'jumpHeight': 0.3, 'jumpLength': [0.0, 0.4, 0.], 'timeStep': 1e-2, 'groundKnots': 20, 'flyingKnots': 20}
# jumping_gait = {'jumpHeight': 0.8, 'jumpLength': [0.0, 0.5, 0.], 'timeStep': 1e-2, 'groundKnots': 20, 'flyingKnots': 30}
ddp = crocoddyl.SolverFDDP(gait.createWalkingProblem(x0, walking_gait['stepLength'], walking_gait['stepHeight'], walking_gait['timeStep'],
                              walking_gait['stepKnots'], walking_gait['supportKnots']))
# ddp = crocoddyl.SolverFDDP(gait.createTrottingProblem(x0, trotting_gait['stepLength'], trotting_gait['stepHeight'], trotting_gait['timeStep'],
#                               trotting_gait['stepKnots'], trotting_gait['supportKnots']))
# ddp = crocoddyl.SolverFDDP(gait.createJumpingProblem(x0, jumping_gait['jumpHeight'], jumping_gait['jumpLength'], jumping_gait['timeStep'],
#                                           jumping_gait['groundKnots'], jumping_gait['flyingKnots']))
print('*** SOLVE A1 WALKING ***')

# Set up the visualizer
cameraTF = [2., 2.68, 0.84, 0.2, 0.62, 0.72, 0.22]
display = crocoddyl.GepettoDisplay(A1, 4, 4, cameraTF, frameNames=[lfFoot, rfFoot, lhFoot, rhFoot])
ddp.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose(), crocoddyl.CallbackDisplay(display)])

xs = [A1.model.defaultState] * (ddp.problem.T + 1)
us = [
    m.quasiStatic(d, A1.model.defaultState)
    for m, d in list(zip(ddp.problem.runningModels, ddp.problem.runningDatas))
]
ddp.solve(xs, us, 100, False, 0.1)

#print(xs)

display = crocoddyl.GepettoDisplay(A1, frameNames=[lfFoot, rfFoot, lhFoot, rhFoot])
print(ddp.forwardPass(0.25))
while True:
    display.displayFromSolver(ddp)
    time.sleep(1.0)
