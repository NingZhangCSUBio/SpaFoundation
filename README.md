# SpaFoundation: A visual foundation model for spatial transcriptomics
Here we present SpaFoundation, a versatile visual foundational model with 80 million trainable parameters, pre-trained on 1.84 million histological image patches to learn general-purpose imaging representations. Using histology images alone, SpaFoundation accurately infers spatial gene expression and demonstrates strong robustness and transferability across tasks such as super-resolution inference, tumor detection, and spatial clustering. 
![Overview](overview.jpg)
## Requirements

- torch == 1.10.0+cu113  
- torchvision == 0.11.0+cu113  
- pip == 23.0  
- pytorch-lightning == 1.6.0  
- matplotlib == 3.6.3  
- scipy == 1.10.1  
- pandas == 1.4.3  
- scprep == 1.2.3  
- timm == 0.9.16  
- ipykernel == 6.29.3  
We recommend creating a new Conda virtual environment with Python 3.8 and setting up the environment according to the above requirements.
For environment management, we recommend [conda](https://docs.conda.io/en/latest/). 


