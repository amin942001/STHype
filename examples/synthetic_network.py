"""

Mycorrhizal network growth and carbon transport models

Murray Shanahan

May 2024

"""

import io

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from PIL import Image

# Dimensions of virtual plate
xmin = -260
xmax = 260
ymin = 0
ymax = 260


# Network occupancy grids
g_res = 1  # grid resolution per unit distance
g_rows = (ymax - ymin) * g_res  # number of grid rows
g_cols = (xmax - xmin) * g_res  # number of grid columns
fungus_grid = np.zeros([g_rows, g_cols], dtype=np.int32)  # fungus occupancy grid
obstacle_grid = np.zeros([g_rows, g_cols], dtype=np.int32)  # obstacle occupancy grid
grids = {"fungus": fungus_grid, "obstacle": obstacle_grid}


# Growth parameters
dir_var = 0.1  # variance in directional change
dldt = 1  # rate of growth
p_extend = 1  # probability that a tip will extend
p_split = dldt * 28 / 100  # probabiity that a tip will split into two tips
split_angle = 0.4  # angle between newly split segments
max_delay_split = 8  # delay after which a node can start to grow
maxnodes = 10000


# ========================
# UTILITIES
# ========================


def barrier(row, gap_start, gap_end):
    """Make an occupancy grid containing a horizontal line with a gap in it."""
    barrier = np.zeros([g_rows, g_cols], dtype=np.int32)  # empty occupancy grid
    for col in range(0, gap_start):  # horizontal line from left edge to gap
        barrier[row, col] = 1
    for col in range(gap_end, g_cols):  # horizontal line from gap to right edge
        barrier[row, col] = 1
    return barrier


def add_obstacles(grids):
    """Add barrier(s) to the virtual plate."""
    obs_grid = grids["obstacle"]
    gap_size = 10
    # # Add a barrier a quarter of the way up the plate
    # row = g_rows // 4
    # gap_start = np.random.randint(0, g_cols - gap_size - 1)
    # gap_end = gap_start + gap_size
    # obs_grid = obs_grid + barrier(row, gap_start, gap_end)
    # Add a barrier half way up the plate
    row = g_rows // 2
    gap_start = np.random.randint(0, g_cols - gap_size - 1)
    gap_end = gap_start + gap_size
    obs_grid = obs_grid + barrier(row, gap_start, gap_end)
    # # Add a barrier 3/4 of the way up the plate
    # row = g_rows * 3 // 4
    # gap_start = np.random.randint(0, g_cols - gap_size - 1)
    # gap_end = gap_start + gap_size
    # obs_grid = obs_grid + barrier(row, gap_start, gap_end)
    grids["obstacle"] = obs_grid
    return grids


def coords_to_grid(coords):
    x = coords[0]
    y = coords[1]
    row = int(np.floor((y - ymin) * g_res))
    col = int(np.floor((x - xmin) * g_res))
    return (row, col)


def in_grid(row, col):
    return row >= 0 and row < g_rows and col >= 0 and col < g_cols


# ========================
# NETWORK GROWTH
# ========================


def add_seg(network, node, hyperedge, end_coords, t):
    """Add a segment from node to a new point with coordinates end_coords."""

    conn = network["conn"]
    conn_hyperedge = network["conn_hyperedge"]
    coords = network["coords"]
    nnodes = network["nnodes"]
    conn[node, nnodes] = t + 1
    conn_hyperedge[node, nnodes] = hyperedge + 1
    coords[nnodes, 0] = end_coords[0]
    coords[nnodes, 1] = end_coords[1]
    nnodes += 1
    network = {
        "conn": conn,
        "coords": coords,
        "nnodes": nnodes,
        "conn_hyperedge": conn_hyperedge,
    }
    return network


def make_self_connection(network, tip_node, hyperedge, old_node, t):
    """Add connection from tip_node to old_node to connectivity matrix."""
    conn = network["conn"]
    conn_hyperedge = network["conn_hyperedge"]
    conn[tip_node, old_node] = t + 1
    conn_hyperedge[tip_node, old_node] = hyperedge + 1
    network["conn"] = conn
    network["conn_hyperedge"] = conn_hyperedge
    return network


def apply_tip_dels(tip_dels, tip_nodes, tip_hyperedges, tip_activations, directions):
    """Remove nodes in tip_dels from tip_nodes and directions."""
    for i in range(len(tip_dels)):
        idx = tip_nodes.index(tip_dels[i])
        tip_nodes.pop(idx)
        tip_hyperedges.pop(idx)
        tip_activations.pop(idx)
        directions.pop(idx)
    return (tip_nodes, tip_hyperedges, tip_activations, directions)


def extend_tip(network, tip_nodes, tip_hyperedges, tip_dels, directions, grids, tip, t):
    """Grow the given tip, adding a new node to the network."""
    tip_node = tip_nodes[tip]
    hyperedge = tip_hyperedges[tip]
    tip_coords = network["coords"][tip_node]
    fungus_grid = grids["fungus"]
    obs_grid = grids["obstacle"]
    # Adjust direction
    old_dir = directions[tip]
    new_dir = old_dir + np.random.normal(0.0, dir_var)
    # Add new node
    new_node_coords = [
        tip_coords[0] + dldt * np.sin(new_dir),
        tip_coords[1] + dldt * np.cos(new_dir),
    ]
    (row, col) = coords_to_grid(new_node_coords)
    if in_grid(row, col) and obs_grid[row, col] == 0:
        if fungus_grid[row, col] == 0:  # check for possible self connection
            new_node_no = network["nnodes"]  # tip's new node no. is nnodes
            tip_nodes[tip] = new_node_no
            network = add_seg(network, tip_node, hyperedge, new_node_coords, t)
            directions[tip] = new_dir
            fungus_grid[row, col] = new_node_no  # update fungus grid
        elif (
            fungus_grid[row, col] != tip_node
        ):  # self-connect if cell occupied by another node
            old_node = fungus_grid[row, col]
            network = make_self_connection(network, tip_node, hyperedge, old_node, t)
            tip_dels.append(tip_node)
    grids["fungus"] = fungus_grid
    return (network, tip_dels, directions, grids)


def split_tip(
    network,
    tip_nodes,
    tip_hyperedges,
    tip_activations,
    tip_dels,
    directions,
    grids,
    tip,
    t,
):
    """Split the given tip into two, adding two new nodes to the network."""
    tip_node = tip_nodes[tip]
    hyperedge = tip_hyperedges[tip]
    tip_coords = network["coords"][tip_node]
    fungus_grid = grids["fungus"]
    obs_grid = grids["obstacle"]
    #  Adjust direction
    old_dir = directions[tip]
    new_dir = old_dir + np.random.normal(0, dir_var)
    # New node extending old tip
    new_node_coords1 = [
        tip_coords[0] + dldt * np.sin(new_dir),
        tip_coords[1] + dldt * np.cos(new_dir),
    ]
    (row, col) = coords_to_grid(new_node_coords1)
    delete = False  # delete tip node flag
    if in_grid(row, col) and obs_grid[row, col] == 0:
        if fungus_grid[row, col] == 0:  # check for self connection
            new_node_no1 = network["nnodes"]  # tip's new node no. is nnodes
            tip_nodes[tip] = new_node_no1
            network = add_seg(network, tip_node, hyperedge, new_node_coords1, t)
            directions[tip] = new_dir
            fungus_grid[row, col] = new_node_no1  # update grid
        elif (
            fungus_grid[row, col] != tip_node
        ):  # self-connect if cell occupied by another node
            old_node = fungus_grid[row, col]
            network = make_self_connection(network, tip_node, hyperedge, old_node, t)
            delete = True  # tip node to be deleted
        else:
            return (network, tip_dels, directions, grids)
    else:
        return (network, tip_dels, directions, grids)
    # New node with new tip
    split_angle_direction = -2 if np.random.rand() > 0.5 else 2
    angle = np.random.normal(split_angle, dir_var)
    new_node_coords2 = [
        tip_coords[0] + dldt * np.sin(new_dir + split_angle_direction * angle),
        tip_coords[1] + dldt * np.cos(new_dir + split_angle_direction * angle),
    ]
    (row, col) = coords_to_grid(new_node_coords2)
    if in_grid(row, col) and obs_grid[row, col] == 0:
        delay = np.random.randint(0, max_delay_split + 1)
        if fungus_grid[row, col] == 0:  # check for self connection
            new_node_no2 = network["nnodes"]  # new tip's node no. is nnodes
            tip_nodes.append(new_node_no2)
            tip_activations.append(t + delay)
            tip_hyperedges.append(new_node_no2)
            network = add_seg(
                network, tip_node, new_node_no2, new_node_coords2, t + delay
            )
            directions.append(new_dir + split_angle_direction * angle)
            fungus_grid[row, col] = new_node_no2  # update grid
        elif (
            fungus_grid[row, col] != tip_node
            and fungus_grid[row, col] != network["nnodes"] - 1
        ):  # self-connect if cell occupied by another node
            old_node = fungus_grid[row, col]
            new_node_no2 = network["nnodes"]  # new tip's node no. is nnodes
            network = make_self_connection(
                network, tip_node, new_node_no2, old_node, t + delay
            )
            # delete = True  # tip node to be deleted
    #  Mark tip node for deletion if eliminated by self-connection
    if delete:
        tip_dels.append(tip_node)
    grids["fungus"] = fungus_grid
    return (network, tip_dels, directions, grids)


def make_trivial_network():
    """Make a small network with a fork in it."""
    conn = np.zeros([maxnodes, maxnodes])  # connectivity matrix
    coords = np.zeros([maxnodes, 2])  # co-ordinates of nodes
    coords[0, 0] = 0.0  # root point is at [0, 0]
    coords[0, 1] = 0.0
    nnodes = 12
    split = 3
    for i in range(nnodes):
        if i <= split:
            coords[i] = [0, i * 0.5]
        elif i % 2 == 1:
            coords[i] = [-1, split * 0.5 + (i - split + 1) // 2 * 0.5]
        else:
            coords[i] = [1, split * 0.5 + (i - split + 1) // 2 * 0.5]
        if i < split:
            conn[i, i + 1] = 1
        elif i == split:
            conn[i, i + 1] = 1
            conn[i, i + 2] = 1
        elif i < nnodes - 2:
            conn[i, i + 2] = 1
    conn[split, split + 1] = 1
    conn[split, split + 2] = 1
    network = network = {"conn": conn, "coords": coords, "nnodes": nnodes}
    tip_nodes = [0, nnodes - 1, nnodes - 2]
    return (network, tip_nodes)


def grow_network(t_growth, grids):
    """Grow a network for t_growth time steps."""
    frames = []  # frames for animation
    conn = np.zeros([maxnodes, maxnodes])  # connectivity matrix
    conn_hyperedge = np.zeros([maxnodes, maxnodes])  # connectivity matrix
    coords = np.zeros([maxnodes, 2])  # co-ordinates of nodes
    coords[0, 0] = 0.0  # root point is at [0, 0]
    coords[0, 1] = 0.0
    nnodes = 1
    network = network = {
        "conn": conn,
        "coords": coords,
        "nnodes": nnodes,
        "conn_hyperedge": conn_hyperedge,
    }
    directions = [0.0]  # prevailing directions of growth for each tip
    tip_nodes = [0]  # the root point is a tip node
    tip_activations = [0]  # the root is there at the beginning
    tip_hyperedges = [0]
    for t in range(t_growth):
        tip_dels = []  # list of tips for deletion
        # Extend the network from each tip
        for tip in range(len(tip_nodes)):
            if tip_activations[tip] > t:
                continue
            if np.random.uniform(0, 1) < p_split:  # split the tip into two
                (network, tip_dels, directions, grids) = split_tip(
                    network,
                    tip_nodes,
                    tip_hyperedges,
                    tip_activations,
                    tip_dels,
                    directions,
                    grids,
                    tip,
                    t,
                )
            elif np.random.uniform(0, 1) < p_extend:  # grow the tip without splitting
                (network, tip_dels, directions, grids) = extend_tip(
                    network,
                    tip_nodes,
                    tip_hyperedges,
                    tip_dels,
                    directions,
                    grids,
                    tip,
                    t,
                )
        # Delete tips that made self-connections
        (tip_nodes, tip_hyperedges, tip_activations, directions) = apply_tip_dels(
            tip_dels, tip_nodes, tip_hyperedges, tip_activations, directions
        )
    #     # Plot the extended network and add to list of frames for animation
    #     frame = plot_network_frame(network, tip_nodes)
    #     frames.append(frame)
    # # Save animation as gif
    # frame_one = frames[0]
    # frame_one.save(
    #     "growth_animation.gif",
    #     format="GIF",
    #     append_images=frames,
    #     save_all=True,
    #     duration=80,
    # )
    return (network, tip_nodes)


# ========================
# CARBON TRANSPORT
# ========================


def initialise_carbon(tip_nodes):
    """Initial carbon loads and surpluses for every node, and statuses for nodes and tips."""
    carbon_load = np.zeros(maxnodes)  # carbon content of each node
    carbon_surplus = np.zeros(maxnodes)  # discounted carbon surplus of each node
    carbon_status = ["neutral" for _ in range(network["nnodes"])]
    # Randomly assign source or sink status to each tip node
    for node in tip_nodes:
        if np.random.uniform(0, 1) < 0.4:
            carbon_status[node] = "source"
            carbon_surplus[node] = 1.0
        else:
            carbon_status[node] = "sink"
            carbon_surplus[node] = -1.0
    # Root node is always a sink
    carbon_status[0] = "sink"
    carbon_surplus[0] = -1.0
    return (carbon_load, carbon_surplus, carbon_status)


def initialise_trivial_network_carbon():
    """Initial carbon loads and surpluses for every node, and statuses for nodes and tips."""
    carbon_load = np.zeros(maxnodes)  # carbon content of each node
    carbon_surplus = np.zeros(maxnodes)  # discounted carbon surplus of each node
    carbon_status = ["neutral" for _ in range(network["nnodes"])]
    # One source, two sinks
    carbon_status[0] = "source"
    carbon_status[10] = "sink"
    carbon_status[11] = "sink"
    carbon_load[0] = 1.0
    carbon_surplus[0] = 1.0
    carbon_surplus[10] = -1.0
    carbon_surplus[11] = -1.0
    # Two sources, one sink
    # carbon_status[0] = 'sink'
    # carbon_status[10] = 'source'
    # carbon_status[11] = 'source'
    # carbon_load[10] = 0.1
    # carbon_load[11] = 0.8
    # carbon_surplus[0] = -1.0
    # carbon_surplus[10] = 1.0
    # carbon_surplus[11] = 1.0
    # Three sources
    # carbon_status[0] = 'source'
    # carbon_status[10] = 'source'
    # carbon_status[11] = 'source'
    # carbon_load[0] = 1.0
    # carbon_load[10] = 1.0
    # carbon_load[11] = 1.0
    # carbon_surplus[0] = 1.0
    # carbon_surplus[10] = 1.0
    # carbon_surplus[11] = 1.0
    return (carbon_load, carbon_surplus, carbon_status)


def update_carbon_surplus(network, carbon_surplus):
    """Spread carbon surplus values through network (one step)."""
    conn = network["conn"]
    nnodes = network["nnodes"]
    conn = conn + np.transpose(conn)  # copy connectivity matrix across diagonal
    old_carbon_surplus = carbon_surplus.copy()
    for node in range(nnodes):
        if carbon_status[node] == "neutral":
            conn_row = conn[node]
            neighbours = np.where(conn_row == 1)[0]
            carbon_surplus[node] = np.mean(old_carbon_surplus[neighbours])
    return carbon_surplus


def update_flow_gradient(network, carbon_surplus):
    """Spread flow gradients through network (one step."""
    conn = network["conn"]
    nnodes = network["nnodes"]
    conn = conn + np.transpose(conn)  # copy connectivity matrix across diagonal
    flow_gradient = np.zeros([maxnodes, maxnodes])
    for node in range(nnodes):
        conn_row = conn[node]
        neighbours = np.where(conn_row == 1)[0]
        neighbours = [
            i for i in neighbours if carbon_surplus[node] > carbon_surplus[i]
        ]  # outgoing flow only
        n_neighbours = len(neighbours)
        for neighbour in neighbours:
            gradient = carbon_surplus[node] - carbon_surplus[neighbour]
            gradient = np.ceil(gradient)
            flow_gradient[node, neighbour] = gradient / n_neighbours
    return flow_gradient


def compute_flow_gradients(network, carbon_surplus):
    """Compute carbon gradients across network."""
    flow_gradient = np.zeros([maxnodes, maxnodes])
    for time in range(50):
        carbon_surplus = update_carbon_surplus(network, carbon_surplus)
    flow_gradient = update_flow_gradient(network, carbon_surplus)
    flow_gradient = np.where(flow_gradient > 0, 1.0, 0.0)  # make either 0 or 1
    nnodes = network["nnodes"]
    return flow_gradient


def update_carbon_load(network, carbon_load, flow_gradient, carbon_status):
    """Compute new carbon loads for every node."""
    epsilon = 1e-10
    max_carbon = 1.0  # carbon capacity per node
    # Remaining carbon capacity per node
    capacity = max_carbon - carbon_load
    # Potential incoming carbon matrix (to each node from each node)
    pot_incoming_m = np.transpose([carbon_load]) * flow_gradient
    # Normalise by capacities of receiving nodes
    capacity_m = capacity * flow_gradient
    sum_capacity = np.transpose([np.sum(capacity_m, axis=1)])
    mask = sum_capacity == 0  # avoid division by zero
    norm_capacity_m = capacity_m / (sum_capacity + mask)
    pot_incoming_m = pot_incoming_m * norm_capacity_m
    # Potential incoming carbon per node
    pot_incoming = np.sum(pot_incoming_m, axis=0)
    # Actual incoming carbon per node
    incoming = np.minimum(pot_incoming, capacity)
    # Normalise potential incoming carbon matrix by actual incoming carbon
    mask = incoming == 0  # avoid division by zero
    norm_pot_incoming_m = pot_incoming_m / (incoming + mask)
    incoming_m = norm_pot_incoming_m * incoming
    # Outgoing carbon
    outgoing_m = -np.transpose(incoming_m)
    outgoing = np.sum(outgoing_m, axis=0)
    carbon_load += incoming + outgoing
    if np.any(carbon_load < -epsilon):
        print("Warning: invalid carbon load(s):")
        print(carbon_load[carbon_load < 0])
    return carbon_load


def replenish(carbon_load, tip_nodes):
    """Maintain high carbon load for sources."""
    for node in tip_nodes:
        if carbon_status[node] == "source":
            if np.random.uniform(0, 1) < 0.05:
                carbon_load[node] = 1.0
            else:
                carbon_load[node] = 0.0
    return carbon_load


def deplete(carbon_load, tip_nodes):
    """Maintain low carbon load for sinks."""
    for node in tip_nodes:
        if carbon_status[node] == "sink" and np.random.uniform(0, 1) < 1.0:
            carbon_load[node] = 0.0
    carbon_load[0] = 0.0  # deplete root node
    return carbon_load


def simulate_carbon_flow(
    t_flow, network, tip_nodes, carbon_load, flow_gradient, carbon_status
):
    # First frame for animation
    frame = plot_carbon_frame(
        network, tip_nodes, carbon_load, flow_gradient, carbon_status, 0
    )
    frames = [frame]
    # Simulate carbon flow
    for time in range(t_flow):
        # Replenish source nodes and deplete sink nodes
        carbon_load = replenish(carbon_load, tip_nodes)
        carbon_load = deplete(carbon_load, tip_nodes)
        # Updates
        carbon_load = update_carbon_load(
            network, carbon_load, flow_gradient, carbon_status
        )
        frame = plot_carbon_frame(
            network, tip_nodes, carbon_load, flow_gradient, carbon_status, time + 1
        )
        frames.append(frame)
    # Save animation as gif
    frame_one = frames[0]
    frame_one.save(
        "carbon_animation.gif",
        format="GIF",
        append_images=frames,
        save_all=True,
        duration=80,
    )


# ========================
# PLOTTING
# ========================


def plot_network(network, tip_nodes):
    conn = network["conn"]
    coords = network["coords"]
    nnodes = network["nnodes"]
    fig, ax = plt.subplots()
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    img = grids["obstacle"]
    ax.matshow(img, extent=[xmin, xmax, ymin, ymax], origin="lower", cmap="Grays")
    for i in range(nnodes):
        xfrom = coords[i, 0]
        yfrom = coords[i, 1]
        for j in range(nnodes):
            if conn[i, j] != 0:
                xto = coords[j, 0]
                yto = coords[j, 1]
                ax.plot([xfrom, xto], [yfrom, yto], color="green", linewidth=1)
    for i in range(len(tip_nodes)):
        node = tip_nodes[i]
        ax.plot(coords[node, 0], coords[node, 1], "ro", markersize=2)
    plt.show()


def plot_network_frame(network, tip_nodes):
    """Plot the fungal network and turn the result into an image."""
    conn = network["conn"]
    coords = network["coords"]
    nnodes = network["nnodes"]
    fig, ax = plt.subplots()
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    img = grids["obstacle"]
    ax.matshow(img, extent=[xmin, xmax, ymin, ymax], origin="lower", cmap="Grays")
    for i in range(nnodes):
        xfrom = coords[i, 0]
        yfrom = coords[i, 1]
        for j in range(nnodes):
            if conn[i, j] != 0:
                xto = coords[j, 0]
                yto = coords[j, 1]
                ax.plot([xfrom, xto], [yfrom, yto], color="green", linewidth=1)
    for i in range(len(tip_nodes)):
        node = tip_nodes[i]
        ax.plot(coords[node, 0], coords[node, 1], "ro", markersize=2)
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format="jpg")
    plt.close()
    frame = Image.open(img_buf)
    return frame


def plot_carbon_frame(
    network, tip_nodes, carbon_load, flow_gradient, carbon_status, time
):
    """Plot carbon loads and turn the result into an image."""
    conn = network["conn"]
    coords = network["coords"]
    nnodes = network["nnodes"]
    fig, ax = plt.subplots()
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    img = grids["obstacle"]
    ax.matshow(img, extent=[xmin, xmax, ymin, ymax], origin="lower", cmap="Grays")
    for i in range(nnodes):
        xfrom = coords[i, 0]
        yfrom = coords[i, 1]
        for j in range(nnodes):
            if conn[i, j] != 0:
                xto = coords[j, 0]
                yto = coords[j, 1]
                ax.plot([xfrom, xto], [yfrom, yto], color="lime", linewidth=1)
        c = Circle(
            coords[i],
            radius=0.2 * carbon_load[i],
            clip_on=False,
            linewidth=1,
            edgecolor="blue",
            facecolor="none",
        )
        ax.add_artist(c)
    for node in tip_nodes:
        if carbon_status[node] == "source":
            ax.plot(coords[node, 0], coords[node, 1], "bo", markersize=3)
        else:
            ax.plot(coords[node, 0], coords[node, 1], "ro", markersize=3)
    ax.set_title("Time is {}".format(time))
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format="jpg")
    plt.close()
    frame = Image.open(img_buf)
    return frame


# ========================
# MAIN
# ========================


print("==============================")
print("Mycorrhizal Network Simulation")
print("Murray Shanahan               ")
print("May 2024                      ")
print("==============================")
print()


# # Grow a network

# print('GROWING NETWORK')
# print()
# grids = add_obstacles(grids)
# maxnodes = 1000
# # (network, tip_nodes) = make_trivial_network()
# (network, tip_nodes) = grow_network(100, grids)
# # plot_network(network, tip_nodes)


# # Simulate carbon flow

# print('SIMULATING CARBON FLOW')
# print()
# # (carbon_load, carbon_surplus, carbon_status) = initialise_trivial_network_carbon()
# (carbon_load, carbon_surplus, carbon_status) = initialise_carbon(tip_nodes)
# flow_gradient = compute_flow_gradients(network, carbon_surplus)
# simulate_carbon_flow(150, network, tip_nodes, carbon_load,  flow_gradient, carbon_status)
