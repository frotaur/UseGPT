# Load and use trained models, easily

What you will need is a '.state' model, which was saved using torchenhanced. Dependencies are the usual pytorch, and you need to install torchenhanced :
```bash
pip install torchenhanced
```

Then, put the `.state`files in the 'models/' folder, and you should be good to go.

See in `test_load.ipynb` for an example of loading a model.


Then, you can test if the loading works using generate.py. Here is how it works : 

```bash
usage: python generate.py [-h] [-d DEVICE] [-g GEN_TOKENS] [-b] [-t TOKENIZER_NAME] [--legacy] models/model_name.state

Generate completions using a specified model.

positional arguments:
  model_path            Path of the save model, in .state format.

options:
  -h, --help            show this help message and exit
  -d DEVICE, --device DEVICE
                        Specify the device to run on, e.g., 'cpu' or 'cuda'. Default is 'cpu'.
  -g GEN_TOKENS, --gen_tokens GEN_TOKENS
                        Number of tokens to generate
  -b, --backward        Use it if it is a backwards model
  -t TOKENIZER_NAME, --tokenizer_name TOKENIZER_NAME
                        Name of the tokenizer to use. Use prefix like 'fr' or 'en'
  --legacy              Use legacy model loading (fast=False) for older models    
```

Example : `python generate.py -d cuda -g 256 -t en --legacy models/en_med.state`

Add/remove the legacy flag if it fails.

If successful, after loading of the model it should start and interface where you can 'prompt' the model, press enter, and it will generate the rest. Press enter with no text to exit.

## Loading models
To load models, you can take a look at test_load.ipynb for an example.

But, in a nutshell, it is very simple :

```python
from torchenhanced import Trainer
from modules import MinGPT

model = Trainer.get_model_from_state(constructor=MinGPT, state_file='modelfile.state')
```

This will load the model (on cpu), and you can use as you please. (Note that this will fail if it's a legacy model, in that case you need to follow the method in test_load.ipynb)

In case you want to save the state_dict of the model as a usual '.pt' file, you can use the utility function from Trainer : 

```python
from torchenhanced import Trainer

model_name, config, weights = Trainer.model_config_from_state(state_path='modefile.state',device='cpu')

```

`model_name` will be the class name of the model (usually MinGPT here)

`config` is a dictionary containing the constructor parameters for the model. In other words, `MinGPT(**config)` will initialize a model with the same architecture as the saved model.

`weights` is a pytorch state_dict with the saved weights of the model. 