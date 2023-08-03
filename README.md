## Technical Coding Challenge - Quantization

Minimize the size footprint of a machine learning model with quantization methods in Pytorch framework. We investigate 3 different quantization techniques: `Dynamic Quantitation`,`Post-Training Static Quantization`,`Quantization-aware training (QAT)` on top of a computer vision ML model, trained with MNIST dataset over an Lenet5 architecture model, and the results depict that even though the accuracy levels remain almost constant, the quantization model sizes are dropped close to 3.3 times for all 3 approaches. 

### Instructions

The code was written and tested on Ubuntu 20.04 operating system, with Python 3.11 installed.
All required packages are located under the `venv` folder. To activate the virtual environment you need to run:

```
pip install virtualenv
virtualenv venv
source venv/bin/activate
```
After activation, you will need to have some useful packages, so run:
```
pip install torch torchvision scikit-learn pandas matplotlib notebook
```
Start locally a Jyputer notebook server as follows:
```python
# Create a kernel based on your virtual environment
python -m ipykernel install --user --name=venv
jupyter notebook
```
Important! Make sure you run the Jupyter local server with the newly created kernel of our virtual environment `venv` and 
run all code blocks included in `technical_coding_challenge.ipynb`.

Alternately, if you do not want to use jupyter notebook option, you just run in your local terminal:
```python
python main.py
```
The results are under the `results` folder.
### References

1. [Pytorch Quantization Documentation](https://pytorch.org/docs/stable/quantization.html). 
2.  [Introduction to Quantization on PyTorch](https://pytorch.org/blog/introduction-to-quantization-on-pytorch/). 
