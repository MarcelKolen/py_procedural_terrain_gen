# procedural terrain generator
This code can be used in three main ways:
	- As a standalone program.
	- As a collection of separate terrain generator components.
	- As an algorithm family producing three different noise maps, a house location map and a road map.

## As a standalone program
To run the program, run it as any other python program with a main function
	python terrain_generator.py [parameters]
This runs the five terrain generation steps and outputs the five steps as images with plotted maps in the running directory.
Run
	python terrain_generator.py help
to get the full list of possible parameters. Each parameter is self-describing.
An example would be
	python terrain_generator.py width=300 height=150 houses=10 max_road_dist=75
This creates a map of 300 by 150, populates it with 10 houses and houses that are no more than 75 apart get a road between them (provided the algorithm doesn’t run out of its budget).

## As a collection of separate terrain generator components
Each component can be used separately. To use the components, import as follows:
	from terrain_generator.py import generate_noise_map, amplify_terrain, smooth_terrain, add_houses, build_roads
In order to plot the maps, import the following module:
	from terrain_generator.py import plot_map
The parameters of each of the 5 components (excluding plot_map) are almost 1 to 1 with the parameters listed by
	python terrain_generator.py help
Only the names are slightly different, but the description is almost identical.
The parameters for plot_map are:
	- noise_map (the map to be plotted)
	- houses
	- roads
	- plot_name
generate_noise_map returns a noise map.
amplify_terrain takes a noise map and returns an amplified noise map.
smooth_terrain takes a noise map and returns an smoothed noise map.
add_houses takes a noise map and returns list of house coordinates as tuples.
build_roads takes a noise map and a list of houses and returns lists of road point coordinates as tuples.

## As an algorithm family producing three different noise maps, a house location map and a road map
This essentially does the same as simply running the script as a python program, however, the result is not five plots, but rather it returns:
	- The initial noise map
	- The amplified noise map (uses the initial noise map as input)
	- The smoothed noise map (uses the amplified noise map as input)
	- A list of house coordinates (uses the smoothed noise map as input)
	- A list of road coordinates (uses the smoothed noise map and the list of house coordinates as inputs)
To use this family of algorithms, or rather to use the automated sequence that calls these algorithms, import the following module:
	from terrain_generator.py import run_base_sequence
The parameters are almost the same as the ones shown in 
	python terrain_generator.py help
only the names are slightly different.

## Closing notes
Play around a bit with all the different settings! The default settings are found to be very satisfactory, but maybe you can find more interesting combinations of parameters ;^)
