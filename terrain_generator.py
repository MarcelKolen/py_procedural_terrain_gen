# Made by Marcel Kolenbrander

import random
import math
import matplotlib.pyplot as plt
import sys

"""
    noise_map Input noise map
    houses List of house coordinates
    roads List of road coordinates
    plot_name Name of the output image
    
    Plots the given map with matplotlib and visualised roads and houses with scatter plots
"""
def plot_map(noise_map, houses=None, roads=None, plot_name='out'):
    # Plot map
    plt.plot()
    
    map_copy = [[i*2 if i >= 0.3 else i for i in j] for j in noise_map]
    
    plt.imshow(map_copy, cmap='viridis', vmin=0, vmax=2)
    plt.colorbar(label='Elevation')
    plt.xlabel('Longitude [x-cord]')
    plt.ylabel('Latitude [y-cord]')
    
    if roads is not None:
        # Plot roads
        for road in roads:
            plt.scatter(*zip(*road), s=40, marker='.', c=[[0.59, 0.59, 0.59],])

    if houses is not None:
        # Plot houses
        plt.scatter(*zip(*houses), s=50, marker='^', c=[[1., 0.19, 0.19],])
    
    plt.savefig(plot_name)
    plt.clf()

"""
    width Sets the width of the map
    height Sets the height of the map
    Returns a noise_map
    
    Generates a procedurally created noise map
"""
def generate_noise_map(width, height):
    print(f"### 1/5:\tGenerating terrain [Width {width}; height {height}] ###")

    # Create a height by width 2d list of zeros
    noise_map = [[0 for i in range(width)] for j in range(height)]

    # Progressively apply variation to the noise map but changing values + or -
    # 5 from the previous entry in the same list, or the average of the
    # previous entry and the entry directly above
    highest_val = 0
    lowest_val = 0
    for y in range(height):
        for x in range(width):
            if x == 0 and y == 0:
                continue
            if y == 0:  # If the current position is in the first row
                new_value = noise_map[y][x - 1] + random.randint(-1000, +1000)
            elif x == 0:  # If the current position is in the first column
                new_value = noise_map[y - 1][x] + random.randint(-1000, +1000)
            else:
                minimum = min(noise_map[y][x - 1], noise_map[y - 1][x])
                maximum = max(noise_map[y][x - 1], noise_map[y - 1][x])
                average_value = (maximum+minimum)/2.0
                new_value = average_value + random.randint(-1000, +1000)
            noise_map[y][x] = new_value
            # check whether value of current position is new top or bottom
            # of range
            if new_value < lowest_val:
                lowest_val = new_value
            elif new_value > highest_val:
                highest_val = new_value

    # Normalises the range, making minimum = 0 and maximum = 1
    difference = float(highest_val - lowest_val)
    print("\n")
    return [[(i - lowest_val)/difference for i in j] for j in noise_map]
    
clamp = lambda a, min_val, max_val: max(min_val, min(a, max_val))
    
"""
    noise_map base noise map to be amplified
    heigh_val Everything from and above this value gets positively amplified
    low_val Everything from and below this value gets negatively amplified
    Returns an amplified noise_map
    
    Amplifies an input noise map to make mountains and bodies of water more distinct
"""
def amplify_terrain(noise_map, heigh_val=0.35, low_val=0.3):
    print(f"### 2/5:\tAmplifying terrain [Heigh value {heigh_val}; Low value {low_val}] ###\n")
    # Split map between two heights:
    #   Everything from heigh_val and above is changed to: node = node + (node-heigh_val)^2
    #     This amplifies higher terrain exponentialy (high mountains get even higher)
    #   Everything from low_val and below is changed to: node = node - node^2
    #     This negatively amplifies lower terrain exponentialy (deep depts get even deeper)
    # The map is clamped between 0. and 1.
    return [[clamp(i + pow(i-heigh_val,2) if i >= heigh_val else (i - pow(i,2) if i <= low_val else i), 0., 1.) for i in j] for j in noise_map]
    

"""
    current_pos Input position
    width Max width of map (bounds checking)
    height Max height of map (bounds checking)
    Returns a list of current_pos' neighbours
    
    Finds all adjacent neighbours
"""
def find_all_neighbours_of_current_smooth_node(current_pos, width, height):
    neighbours = []

    # Look around current position on all eight (8) square locations to find its neighbours 
    # Append all valid neighbours (not out of bounds) to neighbours[]
    # (This could be done in a double forloop iterating over a range of [-1, 0, 1], but this 'unrolled loop' is probably faster (no proof tho))
    neighbour_pos = (current_pos[0] - 1, current_pos[1])
    if neighbour_pos[0] >= 0:
        neighbours.append(neighbour_pos)
    
    neighbour_pos = (current_pos[0], current_pos[1] - 1)
    if neighbour_pos[1] >= 0:
        neighbours.append(neighbour_pos)
    
    neighbour_pos = (current_pos[0] - 1, current_pos[1] - 1)
    if neighbour_pos[0] >= 0 and neighbour_pos[1] >= 0:
        neighbours.append(neighbour_pos)
    
    neighbour_pos = (current_pos[0] + 1, current_pos[1])
    if neighbour_pos[0] < width - 2:
        neighbours.append(neighbour_pos)
    
    neighbour_pos = (current_pos[0], current_pos[1] + 1)
    if neighbour_pos[1] < height:
        neighbours.append(neighbour_pos)
    
    neighbour_pos = (current_pos[0] + 1, current_pos[1] + 1)
    if neighbour_pos[0] < width and neighbour_pos[1] < height:
        neighbours.append(neighbour_pos)
    
    neighbour_pos = (current_pos[0] - 1, current_pos[1] + 1)
    if neighbour_pos[0] >= 0 and neighbour_pos[1] < height:
        neighbours.append(neighbour_pos)
    
    neighbour_pos = (current_pos[0] + 1, current_pos[1] - 1)
    if neighbour_pos[1] >= 0 and neighbour_pos[0] < width:
        neighbours.append(neighbour_pos)
        
    return neighbours
    
"""
    noise_map base noise map to be smoothed
    min_height Everything from and above this value gets smoothend
    max_height Everything up and until this value gets smoothend
    iterations Number of times smoothning is applied to the map
    neighbour_weights Sets the weighted average value for the neighbours of a point. This determined how much neighbours influence the final (height)value of a node
    Returns a smoothed noise_map
    
    Applies weighted average neighbour height values to smoothen a terrain
"""
def smooth_terrain(noise_map, min_height=0., max_height=0.4, iterations=2, neighbour_weights=0.8):
    print(f"### 3/5:\tSmoothning terrain [Iterations {iterations}; min height {min_height}; max height {max_height}; neighbour weights {neighbour_weights}] ###")
    height  = len(noise_map)
    width   = len(noise_map[0])

    map_copy    = [x.copy() for x in noise_map]
    result_map  = None
    
    # Iterate over every node/coordinate in the map
    for i in range(iterations):
        print(f"\tRunning iteration {i + 1} / {iterations}")
        # Smoothning is done by applying values from an old map to a new map
        # Each iteration the maps get "swapped"
        if result_map is not None:
            map_copy = [x.copy() for x in result_map]
        else:
            result_map = [x.copy() for x in map_copy]
        
        for y, hor_map in enumerate(map_copy):
            for x, node in enumerate(hor_map):
                noise_map_val = map_copy[y][x]
                # Only apply smoothning to terrain that falls within the height limits
                if noise_map_val >= min_height and noise_map_val <= max_height: 
                    # Find all 8 neighbours of a node/coordinate
                    nodes = find_all_neighbours_of_current_smooth_node((x,y), width, height)                    
                    sum_of_nodes = noise_map_val
                    nodes_counted = 0
                    # Add each neighbours value to the sum_of_nodes with a weight (this is weighted smoothning where the origin point has a larger influence)
                    for node in nodes:
                        noise_map_val = map_copy[node[1]][node[0]]
                        if noise_map_val >= min_height and noise_map_val <= max_height:
                            sum_of_nodes += noise_map_val * neighbour_weights
                            nodes_counted += 1
                    # Take the weighted average of all neighbours and origine node
                    if nodes_counted > 0:
                        result_map[y][x] = sum_of_nodes / (1 + nodes_counted * neighbour_weights)
    print("\n")
    return result_map
    

set_random_house_locations = lambda width, height: (random.randint(0, width-1), random.randint(0, height-1))
# Added euclid_dist lambda in order to avoid having to use numpy (might not be supported)
euclid_dist = lambda a, b: int(math.sqrt(pow(a[0] - b[0], 2) + pow(a[1] - b[1], 2)))
    
"""
    noise_map Input noise map on which the house locations are going to be based
    number_of_houses Number of houses to be placed on the map (Note: if the restrictions "min_house_distance, min_house_height, max_house_height" 
        are too tight, it might not be able to place all houses)
    min_distance Minimal (eucledian distance (as the bird flies)) between houses
    min_build_height Minimum building height of houses (houses are not spawned at points below this height)
    max_build_height Maximum building height of houses (houses are not spawned at points above this height)
    max_budget If a house cannot be placed (due to above restrictions/too small map), how many time can we try again to find a new spot for a house
    Returns a list of house coordinates
    
    Attempts to populate the map with a set amount of houses according to a few rules
"""
def add_houses(noise_map, number_of_houses=5, min_distance=30, min_build_height=0.3, max_build_height=1., max_budget=10000):
    print(f"### 4/5:\tBuilding houses [houses {number_of_houses}; min distance {min_distance}; min build height {min_build_height}; max build height {max_build_height}; budget {max_budget}] ###")

    height  = len(noise_map)
    width   = len(noise_map[0])
    house_locations = []
    
    for i in range(number_of_houses):
        print(f"\tBuilding house {i + 1} / {number_of_houses}")
    
        house_placement_attempts = 0
    
        # Initial random location of a house
        x_loc_house, y_loc_house = set_random_house_locations(width, height)
        while noise_map[y_loc_house][x_loc_house] < min_build_height or noise_map[y_loc_house][x_loc_house] > max_build_height:
            x_loc_house, y_loc_house = set_random_house_locations(width, height)
        
        if len(house_locations) > 0:
            # Houses already exist
            min_dist_met = False
            # People enjoy their personal space, so houses should not be to close to one and another
            while not min_dist_met and house_placement_attempts <= max_budget:
                min_dist_met = True
                # For every existing house, check whether then new house is far enough away
                for house in house_locations:
                    if euclid_dist(house, (x_loc_house, y_loc_house)) < min_distance:
                        min_dist_met = False
                        x_loc_house, y_loc_house = set_random_house_locations(width, height)
                        while noise_map[y_loc_house][x_loc_house] < min_build_height or noise_map[y_loc_house][x_loc_house] > max_build_height:
                            x_loc_house, y_loc_house = set_random_house_locations(width, height)
                        break
                house_placement_attempts += 1
            if house_placement_attempts > max_budget:
                print("\tCannot place house within budget")
                continue
            # House is not too close to others, so add to array of houses
            house_locations.append([x_loc_house, y_loc_house])
        else:
            # No houses exist yet, so add to array of houses
            house_locations.append([x_loc_house, y_loc_house])
    
    print(f"\t\tPlaced {len(house_locations)}/{number_of_houses} houses")
    
    # If house is on bound of the map, move it away from border (might slightly violate min_distance)
    for house in house_locations:
        if house[1] < 2:
            house[1] = 2
        elif house[1] >= height - 2:
            house[1] = height - 3
            
        if house[0] < 2:
            house[0] = 2
        elif house[0] >= width - 2:
            house[0] = width - 3
    
    houses_as_list_of_tuples = []
    
    # Convert to list of tuples instead of list of lists.
    for house in house_locations:
        houses_as_list_of_tuples.append((house[0], house[1]))
    
    print("\n")
    return houses_as_list_of_tuples

"""
    open_set Set of points from which it can draw from
    f_score Dictionary of points as keys and their f_scores
    Returns the point (item/node) on the map with the lowest f_score out of the current open_set
    
    Looks for the point with the lowest f_score in the open_set
"""
def find_lowest_f_score(open_set, f_score):
    # For each node in the open set find the node with the lowest f_score
    item_with_lowest_f_score = None
    for i in open_set:
        if item_with_lowest_f_score is None:
            item_with_lowest_f_score = i
        elif f_score[i] < f_score[item_with_lowest_f_score]:
            item_with_lowest_f_score = i
    return item_with_lowest_f_score
    
"""
    neighbour_pos Current position of neighbour
    neighbours All neighbours (add neighbour_pos to this object)
    h_score Dictionary of h scores in relation to points on the map
    g_score Dictionary of g scores in relation to points on the map
    f_score Dictionary of f scores in relation to points on the map
    heuristic Heuristic function to determine h_score
    Returns neighbours h_score g_score f_score
    
    Adds a new neighbour to neighbours and applies dynamic programming to the different scores
"""
def add_new_neighbour(neighbour_pos, neighbours, h_score, g_score, f_score, heuristic):
    neighbours.append(neighbour_pos)
    if neighbour_pos not in h_score:
        h_score[neighbour_pos] = heuristic(neighbour_pos)
    if neighbour_pos not in g_score:
        # Any node previously unexplored has an initial g_score of infinity
        g_score[neighbour_pos] = float('inf')
    if neighbour_pos not in f_score:
        # Any node previously unexplored has an initial f_score of infinity
        f_score[neighbour_pos] = float('inf')
        
    return neighbours, h_score, g_score, f_score
    
"""
    current_pos Current position on the map
    h_score Dictionary of h scores in relation to points on the map
    g_score Dictionary of g scores in relation to points on the map
    f_score Dictionary of f scores in relation to points on the map
    heuristic Heuristic function to determine h_score
    width Max width of map (bounds checking)
    height Max height of map (bounds checking)
    Returns neighbours h_score g_score f_score
    
    Looks for all adjacent neighbours arround a given points, precompute their values (if not yet computed) and return all neighbours
"""
def find_all_neighbours_of_current_road_node(current_pos, h_score, g_score, f_score, heuristic, width, height):
    neighbours  = []
    
    # Look around current position on all eight (8) square locations to find its neighbours 
    # Append all valid neighbours (not out of bounds) to neighbours[] and update h,g,f_score dictionaries
    # (This could be done in a double forloop iterating over a range of [-1, 0, 1], but this 'unrolled loop' is probably faster (no proof tho))
    neighbour_pos = (current_pos[0] - 1, current_pos[1])
    if neighbour_pos[0] >= 0:
        neighbours, h_score, g_score, f_score = add_new_neighbour(neighbour_pos, neighbours, h_score, g_score, f_score, heuristic)
    
    neighbour_pos = (current_pos[0], current_pos[1] - 1)
    if neighbour_pos[1] >= 0:
        neighbours, h_score, g_score, f_score = add_new_neighbour(neighbour_pos, neighbours, h_score, g_score, f_score, heuristic)
    
    neighbour_pos = (current_pos[0] - 1, current_pos[1] - 1)
    if neighbour_pos[0] >= 0 and neighbour_pos[1] >= 0:
        neighbours, h_score, g_score, f_score = add_new_neighbour(neighbour_pos, neighbours, h_score, g_score, f_score, heuristic)
    
    neighbour_pos = (current_pos[0] + 1, current_pos[1])
    if neighbour_pos[0] < width:
        neighbours, h_score, g_score, f_score = add_new_neighbour(neighbour_pos, neighbours, h_score, g_score, f_score, heuristic)
    
    neighbour_pos = (current_pos[0], current_pos[1] + 1)
    if neighbour_pos[1] < height:
        neighbours, h_score, g_score, f_score = add_new_neighbour(neighbour_pos, neighbours, h_score, g_score, f_score, heuristic)
    
    neighbour_pos = (current_pos[0] + 1, current_pos[1] + 1)
    if neighbour_pos[0] < width and neighbour_pos[1] < height:
        neighbours, h_score, g_score, f_score = add_new_neighbour(neighbour_pos, neighbours, h_score, g_score, f_score, heuristic)
    
    neighbour_pos = (current_pos[0] - 1, current_pos[1] + 1)
    if neighbour_pos[0] >= 0 and neighbour_pos[1] < height:
        neighbours, h_score, g_score, f_score = add_new_neighbour(neighbour_pos, neighbours, h_score, g_score, f_score, heuristic)
    
    neighbour_pos = (current_pos[0] + 1, current_pos[1] - 1)
    if neighbour_pos[1] >= 0 and neighbour_pos[0] < width:
        neighbours, h_score, g_score, f_score = add_new_neighbour(neighbour_pos, neighbours, h_score, g_score, f_score, heuristic)
    
    return neighbours, h_score, g_score, f_score
    
"""
    noise_map Input noise map to use as terrain guide
    house_locations Houses to build the roads to
    max_distance How far are houses allow to be apart from one and another and still have a road between them. 
        This distance is NOT total road distance, but rather eucledian distance (as the bird flies) between to houses 
        (Note: this value should at least be as high (plus some margin) as "min_house_distance" else there will be no roads! 
        Note2: Larger maps require this value to be higher!)
    max_budget  How long can it take/how many nodes is it allow to explore to find a suitable road between two houses 
        (Note: larger, more complex maps, or larger distances between houses might need a higher budget!)
    edge_cost_mode Selects the version of edge weight calculations
    Returns a coordinates of all road points
    
    Attempts to build roads between houses in accordance with a few rules
"""
def build_roads(noise_map, house_locations, max_distance=50, max_budget=250000, edge_cost_mode="realTerrain"):
    print(f"### 5/5:\tBuilding roads... This could take a little while! [max distance {max_distance}; max budget {max_budget}; edge cost mode \"{edge_cost_mode}\"] ###")

    height  = len(noise_map)
    width   = len(noise_map[0])

    all_roads = []

    # Roads should only lead from one house to another, not to nowhere
    if len(house_locations) < 2:
        print(f"\tou need at least 2 houses to build roads, currently there are {len(house_locations)} houses!")
        return all_roads
        
    if edge_cost_mode == "realTerrain":
        # Weighted edge cost first takes the average of the height on the noise map and then applies an exponential penalty
        # The cheapest routes are arround 0.5 (almost sea level (sea level is 0.3) and no mushy marshes)
        # 0.3 and below is water and the lower the value the deeper the water gets. This would require a bridge (which is expensive)
        # Heigher values indicate mountains (which are also very expensive to build roads on)
        weighted_edge_cost = lambda a, b: pow((4*(((noise_map[a[1]][a[0]] + noise_map[b[1]][b[0]]) / 2) - 0.5)), 4)
    elif edge_cost_mode == "simpleTerrain":
        # The average of the two height values in the noise map, this favors lower terrain for heigher terrain.
        weighted_edge_cost = lambda a, b: (noise_map[a[1]][a[0]] + noise_map[b[1]][b[0]]) / 2
    elif edge_cost_mode == "lowestDifference":
        # Take the absolute difference of the two height values, this tries to avoid large height differences.
        weighted_edge_cost = lambda a, b: abs(noise_map[a[1]][a[0]] - noise_map[b[1]][b[0]])
    elif edge_cost_mode == "flat":
        weighted_edge_cost = lambda a, b: 0
    else:
        print("\tOptions are: realTerrain, simpleTerrain, lowestDifference, flat... Defaulting to realTerrain")
        # See explanation above...
        weighted_edge_cost = lambda a, b: pow((4*(((noise_map[a[1]][a[0]] + noise_map[b[1]][b[0]]) / 2) - 0.5)), 4)
    
    # Iterate over each house and...
    for i, house_a in enumerate(house_locations):
        # ...compare its distance to other houses. Roads should only be build if another house is not too far away (based on euclid_dist)
        for house_b in house_locations[i + 1:]:
            dist = euclid_dist(house_a, house_b)
            # Run A* to find shortest path to neighbouring house with heuristic
            if dist <= max_distance:
                print(f"\tBuilding roads between {house_a} and {house_b}...")
                explored_routes = 0
                
                start       = house_a
                goal        = house_b
                heuristic   = lambda current: euclid_dist(goal, current)
                
                # The initial open set is the starting node
                open_set    = set([start])
                came_from   = dict()
                
                h_score      = {start: heuristic(start)}
                g_score      = {start: 0}
                f_score      = {start: h_score[start]}
                
                # Explore open_set for as long as there is open_set is not empty or until the max_budget has been exceeded
                while len(open_set) > 0 and explored_routes <= max_budget:
                    explored_routes += 1
                    
                    if explored_routes % 5000 == 0:
                        print(f"\t\tSearching: {explored_routes} explored.\tSize current open_set {len(open_set)}...")
                
                    # Look for "best" open route so far (best refers to nodes with the lowest travelcost so far).
                    current = find_lowest_f_score(open_set, f_score)
                    
                    if current == goal:
                        road_map = [current]
                        # Retrace route
                        while current in came_from:
                            current = came_from[current]
                            road_map.insert(0, current)
                        print(f"\t\tFinding this route took {explored_routes} exploration steps.")
                        all_roads.append(road_map)    
                        
                        break
                    elif current is None:
                        break 
                        
                    open_set.remove(current)
                    neighbours, h_score, g_score, f_score = find_all_neighbours_of_current_road_node(current, h_score, g_score, f_score, heuristic, width, height)
                    for neighbour in neighbours:
                        # Tentative gScore is current gScore and the weighted edge cost of the two nodes
                        t_g_score = g_score[current] + weighted_edge_cost(current, neighbour)
                        # If Tentative gScore is better than current gScore of neighbour
                        if t_g_score < g_score[neighbour]:
                            # Walk the bath
                            came_from[neighbour]    = current
                            g_score[neighbour]      = t_g_score
                            f_score[neighbour]      = t_g_score + h_score[neighbour]
                            if neighbour not in open_set:
                                open_set.add(neighbour)
                if explored_routes > max_budget:
                    print(f"\t\tExceeded max_budget of {max_budget} stopping search.")
    print("\n")
    return all_roads

def run_base_sequence(
    _width  = 100,
    _height = 100,
    
    _heigh_val_a    = 0.35,
    _low_val_a      = 0.3,
    
    _min_height_s           = 0.,
    _max_height_s           = 0.4,
    _iterations_s           = 2,
    _neighbour_weights_s    = 0.8,
    
    _number_of_houses_h     = 5,
    _min_distance_h         = 30,
    _min_build_height_h     = 0.3,
    _max_build_height_h     = 1.,
    _max_budget_h           = 10000,
    
    _max_distance_r         = 50,
    _max_budget_r           = 250000,
    _edge_cost_mode_r       = "realTerrain"):
    # Generate map
    _noise_map      = generate_noise_map(_width,_height)
    _noise_map_a    = amplify_terrain(_noise_map, _heigh_val_a, _low_val_a)
    _noise_map_a_s  = smooth_terrain(_noise_map_a,_min_height_s, _max_height_s, _iterations_s, _neighbour_weights_s)
    
    # Add structures
    _houses         = add_houses(_noise_map_a_s, _number_of_houses_h, _min_distance_h, _min_build_height_h, _max_build_height_h, _max_budget_h)
    _roads          = build_roads(_noise_map_a_s, _houses, _max_distance_r, _max_budget_r, _edge_cost_mode_r)
    
    return _noise_map, _noise_map_a, _noise_map_a_s, _houses, _roads
    

def main(args):
    # Default noise map settings
    _width  = 100
    _height = 100
    
    # Default amplify terrain settings
    _heigh_val_a    = 0.35 
    _low_val_a      = 0.3
    
    # Default smooth terrain settings
    _min_height_s           = 0.
    _max_height_s           = 0.4
    _iterations_s           = 2
    _neighbour_weights_s    = 0.8
    
    # Default add houses settings
    _number_of_houses_h     = 5
    _min_distance_h         = 30
    _min_build_height_h     = 0.3
    _max_build_height_h     = 1.
    _max_budget_h           = 10000
    
    # Default build roads settings
    _max_distance_r         = 50
    _max_budget_r           = 250000
    _edge_cost_mode_r       = "realTerrain"
    
    # Default plotting name
    _base_map_name  = 'gen_out'

    # Help text, triggered when the argument help exists. This will stop the program after showing help text
    if 'help' in args:
        print("--!! All of the following parameters are optional, default parameters are set !!--")
        print(f"Usage: python {args[0]} [parameter0 parameter1 parameter2 ...]")
        print("----------------------------------------------------------------------------------")
        print(f"width\tSets the width of the map\tE.g.: width=150\tDefault is {_width}\n")
        print(f"height\tSets the height of the map\tE.g.: height=150\tDefault is {_height}\n")
        print(f"amp_high\tEverything from and above this value gets positively amplified\n\tE.g.: amp_high=0.35\tDefault is {_heigh_val_a}\n")
        print(f"amp_low\tEverything from and below this value gets negatively amplified\n\tE.g.: amp_low=0.30\tDefault is {_low_val_a}\n")
        print(f"smooth_min_height\tEverything from and above this value gets smoothend\n\tE.g.: smooth_min_height=0.1\tDefault is {_min_height_s}\n")
        print(f"smooth_max_height\tEverything up and until this value gets smoothend\n\tE.g.: smooth_max_height=0.45\tDefault is {_max_height_s}\n")
        print(f"smooth_iterations\tNumber of times smoothning is applied to the map\n\tE.g.: smooth_iterations=3\tDefault is {_iterations_s}\n")
        print(f"smooth_weights\tSets the weighted average value for the neighbours of a point. This determined how much neighbours influence the final (height)value of a node\n\tE.g.: smooth_weights=0.8\tDefault is {_neighbour_weights_s}\n")
        print(f"houses\tNumber of houses to be placed on the map (Note: if the restrictions \"min_house_distance, min_house_height, max_house_height\" are too tight, it might not be able to place all houses)\n\tE.g.: houses=10\n\tDefault is {_number_of_houses_h}\n")
        print(f"min_house_distance\tMinimal (eucledian distance (as the bird flies)) between houses\tE.g.: min_house_distance=25\n\tDefault is {_min_distance_h}\n")
        print(f"min_house_height\tMinimum building height of houses (houses are not spawned at points below this height)\tE.g.: min_house_height=0.5\n\tDefault is {_min_build_height_h}\n")
        print(f"max_house_height\tMaximum building height of houses (houses are not spawned at points above this height)\tE.g.: max_house_height=0.9\n\tDefault is {_max_build_height_h}\n")
        print(f"houses_budget\tIf a house cannot be placed (due to above restrictions/too small map), how many time can we try again to find a new spot for a house\n\tE.g.: houses_budget=50000\tDefault is {_max_budget_h}\n")
        print(f"max_road_dist\tHow far are houses allow to be apart from one and another and still have a road between them. This distance is NOT total road distance, but rather eucledian distance (as the bird flies) between to houses (Note: this value should at least be as high (plus some margin) as \"min_house_distance\" else there will be no roads! Note2: Larger maps require this value to be higher!)\n\tE.g.: max_road_dist=75\tDefault is {_max_distance_r}\n")
        print(f"roads_budget\tHow long can it take/how many nodes is it allow to explore to find a suitable road between two houses (Note: larger, more complex maps, or larger distances between houses might need a higher budget!)\n\tE.g.: roads_budget=500000\tDefault is {_max_budget_r}\n")
        print(f"road_mode\tThere are four options:\n\trealTerrain, this is the most complex road builder (avoids difficult terrain),\n\tsimpleTerrain, heigher terrain is more difficult,\n\tlowestDifference, looks at the elevation difference between two nodes (lower difference is better)\n\tflat, flat weight value, almost as the bird flies\n\tE.g.: road_mode=lowestDifference\tDefault is {_edge_cost_mode_r}\n")
        print(f"map_name\tSet name of the generated maps\n\tE.g.: map_name=cool_map\tDefault is {_base_map_name}\n")
        return

    # Convert arguments into dictionary
    args_d = {}
    for arg in args:
        arg_s = arg.split('=')
        if len(arg_s) == 2:
            args_d.update({arg_s[0]:arg_s[1]})
        elif len(arg_s) == 1:
            args_d.update({arg_s[0]:True})
    
    # Parameter processing
    if 'width' in args_d:
        try:
            _width = int(args_d['width'])
        except ValueError:
            print(f"argument 'width' invalid (type \"python {args[0]} help\" for arguments)... Using default")
            
    if 'height' in args_d:
        try:
            _height = int(args_d['height'])
        except ValueError:
            print(f"argument 'height' invalid (type \"python {args[0]} help\" for arguments)... Using default")
            
    if 'amp_high' in args_d:
        try:
            _heigh_val_a = float(args_d['amp_high'])
        except ValueError:
            print(f"argument 'amp_high' invalid (type \"python {args[0]} help\" for arguments)... Using default")
            
    if 'amp_low' in args_d:
        try:
            _low_val_a = float(args_d['amp_low'])
        except ValueError:
            print(f"argument 'amp_low' invalid (type \"python {args[0]} help\" for arguments)... Using default")
            
    if 'smooth_min_height' in args_d:
        try:
            _min_height_s = float(args_d['smooth_min_height'])
        except ValueError:
            print(f"argument 'smooth_min_height' invalid (type \"python {args[0]} help\" for arguments)... Using default")
            
    if 'smooth_max_height' in args_d:
        try:
            _max_height_s = float(args_d['smooth_max_height'])
        except ValueError:
            print(f"argument 'smooth_max_height' invalid (type \"python {args[0]} help\" for arguments)... Using default")
            
    if 'smooth_iterations' in args_d:
        try:
            _iterations_s = int(args_d['smooth_iterations'])
        except ValueError:
            print(f"argument 'smooth_iterations' invalid (type \"python {args[0]} help\" for arguments)... Using default")
            
    if 'smooth_weights' in args_d:
        try:
            _neighbour_weights_s = float(args_d['smooth_weights'])
        except ValueError:
            print(f"argument 'smooth_weights' invalid (type \"python {args[0]} help\" for arguments)... Using default")
            
    if 'houses' in args_d:
        try:
            _number_of_houses_h = int(args_d['houses'])
        except ValueError:
            print(f"argument 'houses' invalid (type \"python {args[0]} help\" for arguments)... Using default")
            
    if 'min_house_distance' in args_d:
        try:
            _min_distance_h = int(args_d['min_house_distance'])
        except ValueError:
            print(f"argument 'min_house_distance' invalid (type \"python {args[0]} help\" for arguments)... Using default")
            
    if 'min_house_height' in args_d:
        try:
            _min_build_height_h = float(args_d['min_house_height'])
        except ValueError:
            print(f"argument 'min_house_height' invalid (type \"python {args[0]} help\" for arguments)... Using default")
            
    if 'max_house_height' in args_d:
        try:
            _max_build_height_h = float(args_d['max_house_height'])
        except ValueError:
            print(f"argument 'max_house_height' invalid (type \"python {args[0]} help\" for arguments)... Using default")
            
    if 'houses_budget' in args_d:
        try:
            _max_budget_h = int(args_d['houses_budget'])
        except ValueError:
            print(f"argument 'houses_budget' invalid (type \"python {args[0]} help\" for arguments)... Using default")
            
    if 'max_road_dist' in args_d:
        try:
            _max_distance_r = int(args_d['max_road_dist'])
        except ValueError:
            print(f"argument 'max_road_dist' invalid (type \"python {args[0]} help\" for arguments)... Using default")
            
    if 'roads_budget' in args_d:
        try:
            _max_budget_r = int(args_d['roads_budget'])
        except ValueError:
            print(f"argument 'roads_budget' invalid (type \"python {args[0]} help\" for arguments)... Using default")
            
    if 'road_mode' in args_d:
        _edge_cost_mode_r = args_d['road_mode']
        
    if 'map_name' in args_d:
        _base_map_name = args_d['map_name']

    # Run base sequence with all parameters to generate map and structures
    _noise_map, _noise_map_a, _noise_map_a_s, _houses, _roads = run_base_sequence(_width, _height, _heigh_val_a, _low_val_a, 
    _min_height_s, _max_height_s, _iterations_s, _neighbour_weights_s, _number_of_houses_h, _min_distance_h, _min_build_height_h, 
    _max_build_height_h, _max_budget_h, _max_distance_r, _max_budget_r, _edge_cost_mode_r)
    
    # Plot the five different results
    print(f"### \tPlotting results with base name \"{_base_map_name}\" ###")
    plot_map(_noise_map, plot_name='0_' + _base_map_name + '_noise_map')
    plot_map(_noise_map_a, plot_name='1_' + _base_map_name + '_amplified_map')
    plot_map(_noise_map_a_s, plot_name='2_' + _base_map_name + '_smoothend_map')
    plot_map(_noise_map_a_s, _houses, plot_name='3_' + _base_map_name + '_with_houses')
    plot_map(_noise_map_a_s, _houses, _roads, plot_name='4_' + _base_map_name + '_with_roads')
    
if __name__ == "__main__":
    main(sys.argv)
