## Core
- The core "library" we have will be turned into an "engine".
- There will be different classes for components of a "robot". for example a robot can consist of a "body" that has the size of 2x2x2(or 2x2 if we think 2D first[it is important to first design in 2D]) and the weight of 20kgs. The body can have a "center of mass" at the center of the body. the robot can have 2 "legs" that are attached to the body. each leg can have 2 "joints" that can rotate. this should be flexible and allow for different configurations.
- Our "engine" with our C++ simulation loop should be able to simulate the robot and allow for different forces to be applied to the robot. The "loss" for the RL system can be defined by the user. Think about the "robot" having a "brain" that is the RL model. 
- The "engine" will have rendering!.

### Rendering

- The engine should have a basic, but usable rendering system. The render can be done with external libraries. But it should be done on the C++ "engine" part. we can have functions like lets say:

```python
robot = fnn.Robot("2LegRobot") # A premade robot for the example.

if __name__ == "__main__":
    while True:
        robot.simulate()
        robot.render()
        fnn.EventManager.process_events() #for exiting the simulation with close events.
    
    fnn.clear()
```

### Loop

- The loop will be done in C++. the Above code is just for you to "get it". 


### Your Questions

- Yes, the differentiable collision will be implemented because we dont want this project to be another gazebo or a robot sim. it is for RL development first and foremost!!

- I dont have a preference for the rendering API. but the rendering in the future will be 3D. that should never be forgotten! We of course dont wanna add a very complex high level rendering API to slow down our building of the core engine. the rendering can be done simply first.

- I dont wanna build a 2 legged robot directly. i wanna lay the foundation that is open to anything!!

- Lets not think about this as an game engine or anything like that. it is still RL focused ML framework that has simulation features for robot simulations. that can be a drone or a 4 legged dog robot, doesnt matter. If this requires major repositioning of the project than it will be done. Tensors should still be there no?




