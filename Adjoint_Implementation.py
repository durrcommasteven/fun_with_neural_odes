import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from functools import reduce
import operator
from typing import Callable, Optional, Union, Tuple, TypeVar
from dataclasses import dataclass


TimeLike = float
PositionLike = Union[np.ndarray, float]
AugmentedType = np.ndarray
PathLike = np.ndarray

T = TypeVar("T")

# Here, our differential equations will be time-independent
DerivFunc = Callable[[T], T]

"""
turn this into a good dataclass
"""


@dataclass(unsafe_hash=True)
class AdjointSensitivityData:
    input_dims: Optional[int] = None

    init_position: Optional[PositionLike] = None
    final_position: Optional[PositionLike] = None
    final_time: Optional[TimeLike] = None

    backwards_adjoint_init_conditions: Optional[AugmentedType] = None

    path: Optional[PathLike] = None
    adjoint_path: Optional[PathLike] = None
    adjoint_param_path: Optional[PathLike] = None

    dl_dz: Optional[PositionLike] = None
    dl_dp: Optional[PositionLike] = None

    def set_path(self, path: PathLike):
        """

        Args:
            path:

        Returns:

        """
        self.path = path
        self.final_position = path[-1]

    def set_augmented_adjoint_path(self, augmented_adjoint_path: PathLike):
        """

        Args:
            augmented_adjoint_path:

        Returns:

        """

        # Extract the adjoint sensitivity path, a(t) = dl_dz(t)
        self.adjoint_path = augmented_adjoint_path[
            :, self.input_dims : 2 * self.input_dims
        ]

        # Extract the parameter adjoint sensitivity path, a_p(t) = dl_dp(t)
        self.adjoint_param_path = augmented_adjoint_path[:, 2 * self.input_dims :]

        # a0 is of the form [z(init_time), dL/dz(init_time), dL/dp]
        # now all but the first 2*z.shape[0] are dL/dp
        self.dl_dp = self.adjoint_param_path[0]

        # the initial adjoint sensitivities are as follows
        self.dl_dz = self.adjoint_path[0]

    def set_backwards_adj_init_condits(
        self,
        final_dl_dz: np.ndarray,
        param_num: int,
    ):
        # put it all together
        # now a_p0 = 0, so we need a zero array with length equal to the number of parameters
        self.backwards_adjoint_init_conditions = np.concatenate(
            list(
                map(
                    self.reshape_to_correct_form,
                    [self.final_position, final_dl_dz, np.zeros(param_num)],
                )
            )
        )

    @staticmethod
    def reshape_to_correct_form(data: Union[np.ndarray, float]):
        """
        Reshape data

        Args:
            data:

        Returns:

        """
        if type(data) == np.ndarray:
            return data.reshape(-1).astype(dtype=np.float32)
        return np.array([data], dtype=np.float32)


class DynamicNN(nn.Module):
    def __init__(self):
        super().__init__()
        # lets try a simple model
        self.f = nn.Linear(2, 2, bias=False)
        self.p_shapes = None

    def forward(self, x):
        return self.f(x)

    def get_jacobian(self, x, out_type="numpy"):
        """
        This is a sort of a misapplication of the
        use of batches. Here we make a batch of
        n copies of the data, one for each
        of the function outputs. We take the gradient
        of each of the outputs, and recombine to get
        the jacobian
        """
        # x should be a torch tensor of shape
        # (len x + len a + len a_p)

        noutputs = x.shape[-1]

        x = x.repeat(noutputs, 1)

        x.requires_grad_(True)
        # y = self.f(x)
        y = self.forward(x)

        y.backward(torch.eye(noutputs))

        if out_type == "numpy":
            return x.grad.data.detach().numpy()
        else:
            return x.grad.data

    def get_parameter_jacobian(self, x, out_type="numpy"):
        """
        I dont know a way to do this rapidly
        lets do a repetition for each output
        """

        noutputs = x.shape[-1]
        param_grads = []

        for out_index in range(noutputs):

            y = self.forward(x)

            # set grads to zero so they dont accumulate
            for p in self.parameters():
                if type(p.grad) != type(None):
                    p.grad.zero_()

            # backprop
            y[out_index].backward()

            # now get grads and flatten them
            param_grads.append(self.flatten_param_grads())

        if out_type == "numpy":
            return torch.stack(param_grads).detach().numpy()
        else:
            return torch.stack(param_grads)

    def flatten_param_grads(self):
        """
        return the grads of the parameters
        which have been systematically and
        replicably flattened
        """
        self.p_shapes = []
        self.flat_params = []

        with torch.no_grad():
            for p in self.parameters():
                self.p_shapes.append(p.size())
                self.flat_params.append(p.grad.flatten())

        return torch.cat(self.flat_params)

    def unflatten_param_grads(self, grads):
        """
        restore the param grads to their original shapes

        set param grads equal to grads

        grads can be either array or tensor
        this converts them to tensors before working with them
        """
        assert len(grads.shape) == 1

        # ensure we have a tensor
        grads = torch.tensor(grads, dtype=torch.float32)

        count = 0
        with torch.no_grad():
            for i, p in enumerate(self.parameters()):

                shape = self.p_shapes[i]
                size = reduce(operator.mul, shape, 1)

                # to be safe, if this has a gradient already,
                # remove it
                if type(p.grad) != type(None):
                    p.grad.zero_()

                # now assign it the appropriate gradient
                p.grad = grads[count : count + size].view(shape)

                count += size
        assert count == len(grads)

    def descend(self, lr):
        """
        descend weights by grads multiplied by dx
        """
        with torch.no_grad():
            for param in self.parameters():
                param -= param.grad * lr


class AdjointMethod:
    def __init__(
        self,
        input_dims: int,
        time_deriv_nn: nn.Module,
        ode_solver: Callable[[TimeLike, PositionLike, DerivFunc], PathLike],
    ):
        """
        This class implement the augmented adjoint sensitivity method

        time_deriv_nn is of the time_deriv_nn class

        loss_function: x-> Reals

        ode_solver is a function of the form
        ode_solver(init_time, final_time, init_position, f, reverse=False)
        """

        self.adjoint_sensitivity_data = AdjointSensitivityData(input_dims=input_dims)

        self.time_deriv_nn = time_deriv_nn
        self.ode_solver = ode_solver

        self.loss_function: Optional[Callable[[torch.Tensor], torch.Tensor]] = None

        self.param_num = sum(
            p.numel() for p in self.time_deriv_nn.parameters() if p.requires_grad
        )

    def dynamical_function(self, position: PositionLike) -> PositionLike:
        """
        Evaluate the time derivative function without tracking gradients.

        Args:
            position: The current position (PositionLike)
            time: The current time (TimeLike)

        Returns:
            gives the time derivative, the current RHS of ODE, PositionLike
        """

        with torch.no_grad():

            return (
                self.time_deriv_nn.forward(torch.from_numpy(position).float())
                .detach()
                .numpy()
            )

    def forward(self, final_time: TimeLike, init_position: PositionLike):
        """
        calculate the forward pass, obtain final_position

        Args:
            final_time:
            init_position:

        Returns:
            final_position
        """

        assert init_position.shape[-1] == self.adjoint_sensitivity_data.input_dims, (
            "init_position has incorrect dimensions. Expected "
            + f"{self.adjoint_sensitivity_data.input_dims}, received {init_position.shape[-1]}"
        )

        self.adjoint_sensitivity_data.final_time = final_time
        self.adjoint_sensitivity_data.init_position = init_position

        self.adjoint_sensitivity_data.set_path(
            self.ode_solver(
                self.adjoint_sensitivity_data.final_time,
                self.adjoint_sensitivity_data.init_position,
                self.dynamical_function,
            )
        )

        return self.adjoint_sensitivity_data.final_position

    def set_loss_func(self, loss_function: Callable[[torch.Tensor], torch.Tensor]):
        """
        target is a numpy array, or pytorch tensor
        loss function is a pytorch function

        set the target, and the loss function
        the loss function is a function of two points (final_position, x2)
        """

        # set loss function
        self.loss_function = loss_function

    def adjoint_dynamical(self, a_aug: AugmentedType) -> AugmentedType:
        """
        here a_aug is given by
        [z, a, a_p]
        and evolves according to
        d[z, a, a_p]/dt = [f, -a f_z, -a f_p]

        When this is implemented, we run it backwards
        Therefore at the end we apply an overall minus
        """

        x = a_aug[: self.adjoint_sensitivity_data.input_dims]
        tensor_x = torch.tensor(x).float()

        # Compute the position derivative
        z_t = self.dynamical_function(x)

        # extract adjoint sensitivity from the augmented vector
        a = a_aug[
            self.adjoint_sensitivity_data.input_dims : 2
            * self.adjoint_sensitivity_data.input_dims
        ]
        assert a.shape[-1] == x.shape[-1]

        # Compute position and parameter derivatives

        # consider standardizing this
        f_z = self.time_deriv_nn.get_jacobian(tensor_x, out_type="numpy")
        af_z = a @ f_z

        # consider standardizing this
        f_p = self.time_deriv_nn.get_parameter_jacobian(tensor_x, out_type="numpy")
        af_p = a @ f_p

        return -np.concatenate([z_t, -af_z, -af_p])

    def set_initial_conditions(self) -> AugmentedType:
        """
        assuming forward has been run, this loads the initial conditions
        for d[a, a_p]/dt = [-a f_z, -a f_p]

        note this is run backwards, from final_time to init_time

        the initial conditions are [dL(z1)/d z1, 0]
        since the loss is defined at L(z1)
        """

        # first obtain a0 = dl/dz1
        final_position_tensor = (
            torch.tensor(self.adjoint_sensitivity_data.final_position)
            .float()
            .requires_grad_()
        )

        # produce gradients for final_position_tensor
        loss = self.loss_function(final_position_tensor)
        loss.backward()

        dl_dz = final_position_tensor.grad.data.detach().numpy()

        assert dl_dz.shape == self.adjoint_sensitivity_data.final_position.shape

        self.adjoint_sensitivity_data.set_backwards_adj_init_condits(
            final_dl_dz=dl_dz,
            param_num=self.param_num,
        )

        return self.adjoint_sensitivity_data.backwards_adjoint_init_conditions

    def adjoint_sensitivity(self, final_time: TimeLike, init_position: PositionLike):
        """
        here we'll enact the adjoint method to determine the gradient
        we'll do this by solving the adjoint differential equation

        we also use initial conditions
        this requires an initial pass of forward to save final_position in memory
        """

        # assert that the target and the loss function
        # have been set
        assert self.loss_function is not None, "Loss function not set"

        # first we do a forward pass
        # this determines the final_position point
        with torch.no_grad():
            self.forward(final_time, init_position)

        # set the initial conditions for the adjoint dynamics
        self.set_initial_conditions()

        # Next solve the adjoint ode, and reverse to get the forward time values

        self.adjoint_sensitivity_data.set_augmented_adjoint_path(
            augmented_adjoint_path=self.ode_solver(
                self.adjoint_sensitivity_data.final_time,
                self.adjoint_sensitivity_data.backwards_adjoint_init_conditions,
                self.adjoint_dynamical,
            )[::-1, :]
        )

        return None

    def descend(self, learning_rate: float):
        """

        Args:
            learning_rate:

        Returns:

        """
        self.time_deriv_nn.unflatten_param_grads(self.adjoint_sensitivity_data.dl_dp)

        # now we descend
        self.time_deriv_nn.descend(learning_rate)

    def zero_grad(self):
        """
        shortcut to clear grads from neural network
        """
        with torch.no_grad():
            for param in self.time_deriv_nn.parameters():
                param.grad.zero_()

    def get_loss(self):
        """
        returns scalar of loss
        assumes forward has been run, and target has been set
        """

        return (
            self.loss_function(
                torch.tensor(self.adjoint_sensitivity_data.final_position).float()
            )
            .detach()
            .numpy()
        )
