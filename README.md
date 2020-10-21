# PrivacyFL: A simulator for privacy-preserving and secure federated learning

This repository contains the source code for running a privacy perserving federated learning simulator. The source code is currently set up for the configuration of three clients performing secure and differentially private federated learning using logistic regresion on the MNIST dataset. This library, however, is meant to be modified so as to simulate your own secure federated machine learning configuration. We hope that this simulation can help users decide whether it is beneficial for them to participate in differentially-private federated learning for a given differentially private algorithm.

## UPDATE : Paper accepted at the 29TH ACM INTERNATIONAL CONFERENCE ON INFORMATION AND KNOWLEDGE MANAGEMENT

Paper and Video Link : https://dl.acm.org/doi/10.1145/3340531.3412771

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.


### Installing
First, clone this repository locally. Then create a conda enviroment by running:
```
conda env create -f environment.yml -n YourEnvironmentName
```
Activate the new enviornment:
```
source activate YourEnvironmentName
```
To validate correct installation `cd` to `src` and run 
```
python run_simulation.py
```
If you encounter any issues, please let us know so that we can help in getting the simulation up and running. 

### Configuring Your Simulation
This library is intended to be modified as needed for your use case. We have provided a default `config.py` file as an example. 

Some simulation behavior can easily be configured by changing the files in the config file.  The file contains Boolean variables `USE_SECURITY` and `USE_DP_PRIVACY` to toggle security and differential privacy features. The security feature does not affect accuracy, however you can set `USE_DP_PRIVACY` to `False` if you want to see what the federated accuracy would be without differential privacy. 

The default `config.py` file also has `USING_CUMULATIVE` set to `True`. What that means is that the dataset for a client on iteration `i` containts all of the datapoints in iteration `i-1` as well as len_per_iteration new datapoints. As such, this flag also makes it so that each client trains its weights from scratch each iteration. Conversely, one can set the `USING_CUMULATIVE` flag to `False`, which will make the dataset non-cumulative and clients perform gradient descent from last iteration's federated weights.

## System Architecure
### Agent 
Agent (defined in agent.py) is the base class for this simulation. It is not meant to be initialized directly, but subclassed to create an agent with whichever behavior you would like. We have provided two sample subclasses, **ClientAgent** and **ServerAgent**, which are both used in the sample simulation. 

### ClientAgent 
An instance of the **ClientAgent** class represents an entity that is training a machine learning model on the same task as the other client agents. The initialization arguments for a client are `agent_number`, `train_datasets`, `evaluator`, `sensitivity`. Client agents are named assigned an `agent_number`, which is then appended to the string `client_agent` to create their name. For example, in the example simulation there are three client agents named `client_agent0`, `client_agent1`, and `client_agent2`. When initialized, clients are also provided their dataset, which in the example is a pyspark dataframe. The client is also passed an `evaluator` which it will use in the simulation to evaluate its own weights and the federated weights. `evaluator`is an instance of the ModelEvaluator class defined in `utils/model_evaluator.py`.
<br/><br/> 
There are two important methods of **ClientAgent** that are invoked by the **ServerAgent**. The first is `compute_weights`, which is called every iteration and prompts the client to perform its machine learning task on its dataset for that iteration. 
The second method is `receive_weights` which is called at the end of every iteration when the server has federated weights to return to the client. 

### ServerAgent 
An instance of the **ServerAgent** class represents a third-party entity that is responsible for invoking the simulation and corresponding with the client agents. It is possible to configure a simulation to use more than one **ServerAgent**, but the straightforward example in the respository currently only creates one instance. Initializaing a **ServerAgent** only requires the same default argument for initializing its superclass **Agent**: `agent_number` which should be `0` for the first server agent. 
<br/><br/> 
**ServerAgent** has one method: `request_values`. The `request_values` is called to signal the server agent to start requesting values from clients, thereby starting the online portion of the simulation. The only argument is `iters` which dictates how many iterations to run the simulation for. Note that if iters is too large, the client agents may run out of data. For the example shown in the repository, only set `iters` to be equal to or less than to the `iterations` argument in the **Initializer** `__init__` method, since that is the method that creates the datasets and distributes them to the clients. If you wish to change the behavior of the simulation, `request_values` is a good place to start and subsequently add/modify any methods that are called in the **ClientAgent** class. 
<br/><br/> 
The `request_values` method first requests weights in parallel from the clients by calling their `compute_weights` method, averages them, and then returns them to the clients in parallel by calling their `receive_weights` method. In the current example, the client returns a message to the server agent through the `receive_weights` method indicating whether its weights have converged. 

### Initializer 
An instance of the **Initializer** class is used to initialize the agents and model evaluator. In addition, any offline stage logic, such as a diffie-helman key exchange, should occur in this class. In our example, it loads the MNIST dataset and processes it for the client agent instances. 
<br/><br/> 
To commence the simulation, the initializer's `run_simulation` method is invoked, which then invokes the server agent's `request_values` method.
### Directory
The **Directory** class contains a mapping of agent names to agent instances that allows agents to invoke other agents' methods by only having their name. An instance of **Directory** is created in the `__init__` method of the **Initializer** class after all the agents have been created. It is then passed on to all the agents using their `set_directory` method.

An example usage to call some method of client_agent1 would be:
`self.directory.clients['client_agent1'].METHOD_TO_CALL()'`

### Utils Folder
This folder contains utilities such as data processing, differential privacy functions, and more. For the most part, functions in here are implementation specific and you should feel free to add any auxiliary functions scripts. 

## Features 
The library is intended to help users simulate secure federated learning to decide whether it would be feasible and beneficial. A snippet of a sample output looks like:
```
Performance Metrics for client_agent2 on iteration 1 
------------------------------------------- 
Personal accuracy: 0.8283333333333334 
Personal computation time: 0:00:01.194242 
Federated accuracy: 0.8566666666666667 
Simulated time to receive federated weights: 0:00:07.202375 

Performance Metrics for client_agent0 on iteration 1 
------------------------------------------- 
Personal accuracy: 0.8216666666666667 
Personal computation time: 0:00:01.198737 
Federated accuracy: 0.8566666666666667 
Simulated time to receive federated weights: 0:00:09.202375 
```
As you can see, the simulation prints out i) the personal accuracy: the accuracy that the client can obtain by itself on its own dataset, adding no DP noise. NOTE: this quantitiy does incorporate other client's data if you set the config.USING_CUMULATIVE flag to False, since that indicates to clients that they should start training on this iteration using the federated weights from the previous iteration since the datasets aren't cumulative. ii) the federated accuracy: the accuracy of the federated model which is the average of all the clients' personal weights + differentially private noise for that iteration. Note that while the clients benefit from participating in the simulation in this example, that is not always the case. In particular, as one increases the amount of differentially private noise, the federated accuracy is expected to decrease. On the other hand, the personal accuracy will remain the same since it is assumed you don't add differentially private noise to your personal model since you are not sharing it. 
<br/><br/> 
In addition, this library also allows you to simulate how long it would take to receive the federated values back for each iteration. `Personal computation time` indicates how long your training took for that iteration while `Simulated time to receive federated weights` takes into account user-defined communication latencies between the clients and the server, as well as how long it took the other clients to compute their weights and the server to average them.

## Authors

Vaikkunth Mugunthan* Anton Peraire* Lalana Kagal

## License

This project is licensed under the MIT License 

MIT License

Copyright (c) 2020 Vaikkunth, Anton, Lalana (PrivacyFL)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
