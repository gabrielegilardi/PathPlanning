"""
Metaheuristic Minimization Using Particle Swarm Optimization.

Copyright (c) 2021 Gabriele Gilardi
"""

import numpy as np


def PSO(func, LB, UB, nPop=40, epochs=500, K=0, phi=2.05, vel_fact=0.5,
        conf_type='RB', IntVar=None, normalize=False, rad=0.1, args=[],
        Xinit=None):
    """
    func            Function to minimize
    LB              Lower boundaries of the search space
    UB              Upper boundaries of the search space
    nPop            Number of agents (population)
    epochs          Number of iterations
    K               Average size of each agent's group of informants
    phi             Coefficient to calculate the two confidence coefficients
    vel_fact        Velocity factor to calculate the maximum and the minimum
                    allowed velocities
    conf_type       Confinement type (on the velocities)
    IntVar          List of indexes specifying which variable should be treated
                    as integers
    normalize       Specifies if the search space should be normalized (to
                    improve convergency)
    rad             Normalized radius of the hypersphere centered on the best
                    particle
    args            Tuple containing any parameter that needs to be passed to
                    the function
    Xinit           Initial position of each agent

    Dimensions:
    (nVar, )        LB, UB, LB_orig, UB_orig, vel_max, vel_min, swarm_best_pos
                    Xinit
    (nPop, nVar)    agent_pos, agent_vel, agent_best_pos, Gr, group_best_pos,
                    agent_pos_orig, agent_pos_tmp, vel_conf, out, x_sphere, u
    (nPop, nPop)    informants, informants_cost
    (nPop)          agent_best_cost, agent_cost, p_equal_g, better, r_max, r,
                    norm
    (0-nVar, )      IntVar
    """
    # Dimension of the search space and max. allowed velocities
    nVar = len(LB)
    vel_max = vel_fact * (UB - LB)
    vel_min = - vel_max

    # Confidence coefficients
    w = 1.0 / (phi - 1.0 + np.sqrt(phi**2 - 2.0 * phi))
    cmax = w * phi

    # Probability an agent is an informant
    p_informant = 1.0 - (1.0 - 1.0 / float(nPop)) ** K

    # Normalize search space
    if (normalize):
        LB_orig = LB.copy()
        UB_orig = UB.copy()
        LB = np.zeros(nVar)
        UB = np.ones(nVar)

    # Define (if any) which variables are treated as integers (indexes are in
    # the range 1 to nVar)
    if (IntVar is None):
        nIntVar = 0
    elif (IntVar == 'all'):
        IntVar = np.arange(nVar, dtype=int)
        nIntVar = nVar
    else:
        IntVar = np.asarray(IntVar, dtype=int) - 1
        nIntVar = len(IntVar)

    # Initial position of each agent
    if (Xinit is None):
        agent_pos = LB + np.random.rand(nPop, nVar) * (UB - LB)
    else:
        Xinit = np.tile(Xinit, (nPop, 1))
        if (normalize):
            agent_pos = (Xinit - LB_orig) / (UB_orig - LB_orig)
        else:
            agent_pos = Xinit

    # Initial velocity of each agent (with velocity limits)
    agent_vel = (LB - agent_pos) + np.random.rand(nPop, nVar) * (UB - LB)
    agent_vel = np.fmin(np.fmax(agent_vel, vel_min), vel_max)

    # Initial cost of each agent
    if (normalize):
        agent_pos_orig = LB_orig + agent_pos * (UB_orig - LB_orig)
        agent_cost = func(agent_pos_orig, args)
    else:
        agent_cost = func(agent_pos, args)

    # Initial best position/cost of each agent
    agent_best_pos = agent_pos.copy()
    agent_best_cost = agent_cost.copy()

    # Initial best position/cost of the swarm
    idx = np.argmin(agent_best_cost)
    swarm_best_pos = agent_best_pos[idx, :]
    swarm_best_cost = agent_best_cost[idx]
    swarm_best_idx = idx

    # Initial best position of each agent using the swarm
    if (K == 0):
        group_best_pos = np.tile(swarm_best_pos, (nPop, 1))
        p_equal_g = \
            (np.where(np.arange(nPop) == idx, 0.75, 1.0)).reshape(nPop, 1)

    # Initial best position of each agent using informants
    else:
        informants = np.where(np.random.rand(nPop, nPop) < p_informant, 1, 0)
        np.fill_diagonal(informants, 1)
        group_best_pos, p_equal_g = group_best(informants, agent_best_pos,
                                               agent_best_cost)

    # Main loop
    for epoch in range(epochs):

        # Determine the updated velocity for each agent
        Gr = agent_pos + (1.0 / 3.0) * cmax * \
             (agent_best_pos + group_best_pos - 2.0 * agent_pos) * p_equal_g
        x_sphere = hypersphere_point(Gr, agent_pos)
        agent_vel = w * agent_vel + Gr + x_sphere - agent_pos

        # Impose velocity limits
        agent_vel = np.fmin(np.fmax(agent_vel, vel_min), vel_max)

        # Temporarly update the position of each agent to check if it is
        # outside the search space
        agent_pos_tmp = agent_pos + agent_vel
        if (nIntVar > 0):
            agent_pos_tmp[:, IntVar] = np.round(agent_pos_tmp[:, IntVar])
        out = np.logical_not((agent_pos_tmp > LB) * (agent_pos_tmp < UB))

        # Apply velocity confinement rules
        if (conf_type == 'RB'):
            vel_conf = random_back_conf(agent_vel)

        elif (conf_type == 'HY'):
            vel_conf = hyperbolic_conf(agent_pos, agent_vel, UB, LB)

        elif (conf_type == 'MX'):
            vel_conf = mixed_conf(agent_pos, agent_vel, UB, LB)

        # Update velocity and position of each agent (all <vel_conf> velocities
        # are smaller than the max. allowed velocity)
        agent_vel = np.where(out, vel_conf, agent_vel)
        agent_pos += agent_vel
        if (nIntVar > 0):
            agent_pos[:, IntVar] = np.round(agent_pos[:, IntVar])

        # Apply position confinement rules to agents outside the search space
        agent_pos = np.fmin(np.fmax(agent_pos, LB), UB)
        if (nIntVar > 0):
            agent_pos[:, IntVar] = np.fmax(agent_pos[:, IntVar],
                                           np.ceil(LB[IntVar]))
            agent_pos[:, IntVar] = np.fmin(agent_pos[:, IntVar],
                                           np.floor(UB[IntVar]))

        # Calculate new cost of each agent
        if (normalize):
            agent_pos_orig = LB_orig + agent_pos * (UB_orig - LB_orig)
            agent_cost = func(agent_pos_orig, args)
        else:
            agent_cost = func(agent_pos, args)

        # Update best position/cost of each agent
        better = (agent_cost < agent_best_cost)
        agent_best_pos[better, :] = agent_pos[better, :]
        agent_best_cost[better] = agent_cost[better]

        # Update best position/cost of the swarm
        idx = np.argmin(agent_best_cost)
        if (agent_best_cost[idx] < swarm_best_cost):
            swarm_best_pos = agent_best_pos[idx, :]
            swarm_best_cost = agent_best_cost[idx]
            swarm_best_idx = idx

        # If the best cost of the swarm did not improve ....
        else:
            # .... when using swarm -> do nothing
            if (K == 0):
                pass

            # .... when using informants -> change informant groups
            else:
                informants = \
                    np.where(np.random.rand(nPop, nPop) < p_informant, 1, 0)
                np.fill_diagonal(informants, 1)

        # Update best position of each agent using the swarm
        if (K == 0):
            group_best_pos = np.tile(swarm_best_pos, (nPop, 1))

        # Update best position of each agent using informants
        else:
            group_best_pos, p_equal_g, = group_best(informants, agent_best_pos,
                                                    agent_best_cost)

    # If necessary de-normalize and determine the (normalized) distance between
    # the best particle and all the others
    if (normalize):
        delta = agent_best_pos - swarm_best_pos         # (UB-LB = 1)
        swarm_best_pos = LB_orig + swarm_best_pos * (UB_orig - LB_orig)
    else:
        deltaB = np.fmax(UB-LB, 1.e-10)             # To avoid /0 when LB = UB
        delta = (agent_best_pos - swarm_best_pos) / deltaB

    # Number of particles in the hypersphere of radius <rad> around the best
    # particle
    dist = np.linalg.norm(delta/np.sqrt(nPop), axis=1)
    in_rad = (dist < rad).sum()

    # Return info about the solution
    info = (swarm_best_cost, swarm_best_idx, in_rad)

    return swarm_best_pos, info


def group_best(informants, agent_best_pos, agent_best_cost):
    """
    Determines the group best position of each agent based on the agent
    informants.
    """
    nPop, nVar = agent_best_pos.shape

    # Determine the cost of each agent in each group (set to infinity the value
    # for agents that are not informants of the group)
    informants_cost = np.where(informants == 1, agent_best_cost, np.inf)

    # For each agent determine the agent with the best cost in the group and
    # assign its position to it
    idx = np.argmin(informants_cost, axis=1)
    group_best_pos = agent_best_pos[idx, :]

    # Build the vector to correct the velocity update for the corner case where
    # the agent is also the group best
    p_equal_g = (np.where(np.arange(nPop) == idx, 0.75, 1.0)).reshape(nPop, 1)

    return group_best_pos, p_equal_g


def hypersphere_point(Gr, agent_pos):
    """
    For each agent determines a random point inside the hypersphere (Gr,|Gr-X|),
    where Gr is its center, |Gr-X| is its radius, and X is the agent position.
    """
    nPop, nVar = agent_pos.shape

    # Hypersphere radius of each agent
    r_max = np.linalg.norm(Gr - agent_pos, axis=1)

    # Randomly pick a direction using a normal distribution and a radius
    # (inside the hypersphere)
    u = np.random.normal(0.0, 1.0, (nPop, nVar))
    norm = np.linalg.norm(u, axis=1)
    r = np.random.uniform(0.0, r_max, nPop)

    # Coordinates of the point with direction <u> and at distance <r> from the
    # hypersphere center
    x_sphere = u * (r / norm).reshape(nPop, 1)

    return x_sphere


def hyperbolic_conf(agent_pos, agent_vel, UB, LB):
    """
    Applies hyperbolic confinement to agent velocities (calculation is done on
    all agents to avoid loops but the change will be applied only to the agents
    actually outside the search space).
    """
    # If the update velocity is > 0
    if_pos_vel = agent_vel / (1.0 + np.abs(agent_vel / (UB - agent_pos)))

    # If the update velocity is <= 0
    if_neg_vel = agent_vel / (1.0 + np.abs(agent_vel / (agent_pos - LB)))

    # Confinement velocity
    vel_conf = np.where(agent_vel > 0, if_pos_vel, if_neg_vel)

    return vel_conf


def random_back_conf(agent_vel):
    """
    Applies random-back confinement to agent velocities (calculation is done on
    all agents to avoid loops but the change will be applied only to the agents
    actually outside the search space).
    """
    nPop, nVar = agent_vel.shape

    # Confinement velocity
    vel_conf = - np.random.rand(nPop, nVar) * agent_vel

    return vel_conf


def mixed_conf(agent_pos, agent_vel, UB, LB):
    """
    Applies a mixed-type confinement to agent velocities (calculation is done on
    all agents to avoid loops but the change will be applied only to the agents
    actually outside the search space).

    For each agent the confinement type (hyperbolic or random-back) is choosen
    randomly.
    """
    nPop, nVar = agent_pos.shape

    # Hyperbolic confinement
    vel_conf_HY = hyperbolic_conf(agent_pos, agent_vel, UB, LB)

    # random-back confinement
    vel_conf_RB = random_back_conf(agent_vel)

    # Confinement velocity
    gamma = np.random.rand(nPop, nVar)
    vel_conf = np.where(gamma >= 0.5, vel_conf_HY, vel_conf_RB)

    return vel_conf
