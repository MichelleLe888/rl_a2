# RL_A2
This project contains the implementation of Deep Q learning
algorithm applied to the openAI CartPole-v1 Gym environment. Project explored the significance of the core elements of
the algorithm: Experience Buffer and Target Network.
Versions of the packages used:\
tensorflow 2.4.1\
gym 0.21.0\
There are 2 parts of the project:
1. you can run one iteration of the Deep Q-learning and specify the component to use with the default
hyper-parameters values:

To include elements mention them in the command line arguments in the
following manner:

To include both: \
python DQL.py target_network experience_buffer \
To include none:\
python DQL.py\
To include one:\
python DQL.py target_network\
python DQL.py experience_buffer

DQL.py file performs a single experiment with default hyper-parameters values.

2. There is a suit of experiments to explore the significance of single
hyper-parameter. Experimentation include:
a) learning rate
b) different action selection schemes: epsilon greedy, annealing epsilon and boltzmann softmax
b) exploration parameter epsilon for epsilon greedy policy
c) exploration parameter temperature for Boltzmann action selection
d) number of nodes in the neural network based models of the agent
e) memory buffer size
f) testing the siginificance of Target Network and Experience buffer:
ablation study that turns those components off and on.
Experiments can only be run separately and the rest of the hyper-parameters values except the one
that was chosen for the exploration are fixed as a default values. To choose the specified
experiement you must use the following command:\
a) python3 Exploration.py --experiment_name=learning_rate\
b) python3 Exploration.py --experiment_name=policy\
c) python3 Exploration.py --experiment_name=Boltz\
d) python3 Exploration.py --experiment_name=architecture\
e) python3 Exploration.py --experiment_name=memory_buffer\
f) python3 Exploration.py --experiment_name=elements\
