A package is a folder containing multiple `.py` files (Modules) like torch.nn is a package and torch.nn.functional is a specific module

A module is an entire file and it can hold many independent functions, variable and classes 

```
mini_dl/                  <-- Package (Folder)
│├── __init__.py          <-- Initializer (Makes folder a package)
│├── layers.py            <-- Module (File containing classes)
└── utils.py              <-- Module (File containing functions/variables)
```

```python
import mini_dl.utils

# Because of our __init__.py having `from .layers import Linear`, we don't have to write 'mini_dl.layers.Linear'
layer_instance = mini_dl.Linear(in_features=2, out_features=1)

# Because we implemented __call__, we don't need to write 'layer_instance.forward(10)'
# We can call the object directly like a function!
output = layer_instance(10)  
print("Final output:", output)
```