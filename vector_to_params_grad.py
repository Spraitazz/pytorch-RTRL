import torch

def add_vector_to_parameters_grad(grad_vec, parameters):
    r"""Convert one vector to the parameters' gradients

    Arguments:
        vec (Tensor): a single vector represents the parameters of a model.
        parameters (Iterable[Tensor]): an iterator of Tensors that are the
            parameters of a model.
    """
    # Ensure vec of type Tensor
    if not isinstance(grad_vec, torch.Tensor):
        raise TypeError('expected torch.Tensor, but got: {}'
                        .format(torch.typename(grad_vec)))
    # Flag for the device where the parameter is located
    param_device = None

    # Pointer for slicing the vector for each parameter
    pointer = 0
    for param in parameters:
        # Ensure the parameters are located in the same device
        param_device =  torch.nn.utils.convert_parameters._check_param_device(param, param_device)

        # The length of the parameter
        num_param = param.numel()
        # Slice the vector, reshape it, and replace the old data of the parameter
        #if param.grad is None:
            #param.grad.data = grad_vec[pointer:pointer + num_param].view_as(param.grad).data
        #else:
        param.grad.data = param.grad.data + grad_vec[pointer:pointer + num_param].view_as(param.grad).data

        # Increment the pointer
        pointer += num_param
