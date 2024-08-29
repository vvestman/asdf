## Creating custom network architectures

To create a custom neural network architecture, create a new class that extends the `BaseNet` in [asdf/src/networks/architectures.py](asdf/src/networks/architectures.py).

To execute experiments using your custom network, change the network class in the `run_configs.py` of your recipe. For example in your run configs you could specify:
``` txt
network.network_class = 'asdf.src.networks.myarchitectures.MyCustomClass'
```
