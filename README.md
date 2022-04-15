# self-driving-2D-NEAT
This repository contains the Python code for creating a self-driving AI based on the NeuroEvolution of Augmenting Topologies algorithm. NEAT is used to evolve Artificial Neural Networks using genetic algorithmic techniques.

* Each generation has 20 instances.
* Each instance is associated with a feed-forward neural network which learns how to navigate the track.
* Each instance is given a fitness value according to the how long it remained on the track.
* New generations are created until at least one car is able to finish a lap without leaving the track.


## Packages Required
1. pygame

          pip install pygame

2. neat-python

          pip install neat-python==0.92

## The Car
The car has a total of 3 sensors, positioned at -45, 0 and 45 degrees respectively. Each sensor has a maximum range of 200 pixels. The blue circles are used to detect if the car left the track (by detecting the green color of grass around the track). 

![alt text](https://github.com/AAnirudh07/self-driving-2D-NEAT/blob/main/car_with_sensors.JPG "Car with sensors")



## The Network
A simple feed forward neural network with 3 inputs and 2 outputs is used. There are **no hidden layers**. The 3 inputs correspond to the sensor values. The were orignally four outputs: Left, Right, Speed up, Slow down. However, after some experimentation it was found that the later two increased the complexity of the network unnecessarily. The final network has only two outputs: Left and Right. 


## Tunable Parameters
The following parameters can be modified:
1. config.txt
          
          fitness_criterion
          fitness_threshold
          reset_on_extinction
          pop_size
          activation_default
          num_hidden
          num_inputs 
          num_outputs
          
2. main.py

          angle
          rotation
          velocity
          radar_angles
          translation constant in drive() method of Car class
