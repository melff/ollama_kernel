from ipykernel.kernelapp import IPKernelApp
from . import OllamaKernel

IPKernelApp.launch_instance(kernel_class=OllamaKernel)
