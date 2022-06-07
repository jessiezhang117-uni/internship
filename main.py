from controller import Controller
import math

PI = 3.1415926

model = Controller()
target = [0, 0, 0, 0, 0, 0, 0]
model.move_group_to_joint_target('Arm', target, plot=True)
model.stay(3000)

#def sin_cal(A, w, t):
    #return A * math.sin(w * t)

#trajectory = []
#precision = 0.001

#A = []
#w = []
#for i in range(7):
    #A.append((8 - i) * 0.2)
    #w.append((i + 1) * 0.5)

#for i in range(10000):
    #tmp = []
    #for k in range(7):
        #tmp.append([sin_cal(A[k], w[k], i * precision)])
    #trajectory.append(tmp)

#result = model.move_group_along_trajectory(group='Arm', target=trajectory, plot=True)
#print("result: {}".format(result))
#model.stay(5000)
