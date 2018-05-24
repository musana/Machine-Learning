import random as r 
import time
import numpy as np

min_, max_, rows, d = -4.5, 4.5, 5, 2
c1, c2, w = 2, 2, 1.2
vmax = 4

swarm = np.random.uniform(min_, max_, (rows, d))
velocity = np.zeros((rows, d))

def z(part):
    return (((1.5 - part[0]) + (part[0]*part[1]))**2) + (((2.25 - part[0]) + (part[0]*(part[1]**2)))**2) + (((2.265 - part[0]) + (part[0]*(part[1]**3)))**2)

ftns = np.array(list(map(z, swarm)))
pbestValue    = ftns
pbestPosition = swarm    
gbestValue    = min(ftns)
gbestPosition = swarm[np.where(ftns == min(ftns))[0][0]]

for d in range(1000):
    for i, particle in enumerate(swarm):
        velocity[i] = ((w * velocity[i]) + (c1*r.random() * (pbestPosition[i] - particle)) + (c2*r.random() * (gbestPosition - particle)))

    velocity[velocity > vmax]  = vmax
    velocity[velocity < -vmax] = -vmax
    swarm = swarm + velocity
    swarm[swarm > max_] = max_
    swarm[swarm < min_] = min_

    for index, particle in enumerate(swarm):
        if z(particle) < pbestValue[index]:
            pbestValue[index]    = z(particle)
            pbestPosition[index] = particle

    for index, particle in enumerate(pbestPosition):
        if min(pbestValue) < gbestValue:
            gbestValue = min(pbestValue)
            gbestPosition = particle
    time.sleep(.1)
    print(d,gbestValue,gbestPosition)