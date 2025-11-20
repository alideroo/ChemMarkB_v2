import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChebyKANLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        degree: int = 5,
        scale_base: float = 1.0,
        scale_cheby: float = 1.0,
        base_activation: type[nn.Module] = nn.SiLU,
        use_bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.degree = degree
        self.scale_base = scale_base
        self.scale_cheby = scale_cheby
        self.base_activation = base_activation()
        self.use_bias = use_bias

        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        
        self.cheby_coeffs = nn.Parameter(
            torch.Tensor(out_features, in_features, degree + 1)
        )

        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)

        self.register_buffer(
            "cheby_orders", 
            torch.arange(0, degree + 1).float()
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(
            self.base_weight, 
            a=math.sqrt(5) * self.scale_base
        )

        with torch.no_grad():
            std = self.scale_cheby / math.sqrt(self.in_features)
            self.cheby_coeffs.normal_(0, std)

        if self.use_bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.base_weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.normal_(self.bias, 0, bound)

    def chebyshev_polynomials(self, x: torch.Tensor) -> torch.Tensor:

        x = torch.tanh(x.clamp(-1, 1))
        
        theta = torch.acos(x)
        
        theta_n = theta.unsqueeze(-1) * self.cheby_orders
        T_n = torch.cos(theta_n) 
        
        return T_n

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        original_shape = x.shape
        x = x.view(-1, self.in_features)

        base_output = F.linear(self.base_activation(x), self.base_weight)

        T_n = self.chebyshev_polynomials(x) 
        cheby_output = torch.einsum('bik,oik->bo', T_n, self.cheby_coeffs)

        output = base_output + cheby_output

        if self.use_bias:
            output += self.bias

        output = output.view(*original_shape[:-1], self.out_features)
        
        return output

    def regularization_loss(self, regularize_coeffs: float = 1.0) -> torch.Tensor:

        coeffs_l2 = self.cheby_coeffs.pow(2).mean()
        return regularize_coeffs * coeffs_l2


class ChebyKAN(nn.Module):

    def __init__(
        self,
        layers_hidden: list[int],
        degree: int = 7,
        scale_base: float = 0.5,
        scale_cheby: float = 0.3,
        base_activation: type[nn.Module] = nn.SiLU,
        use_bias: bool = False,
    ):
        super().__init__()
        
        self.layers = nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                ChebyKANLinear(
                    in_features,
                    out_features,
                    degree=degree,
                    scale_base=scale_base,
                    scale_cheby=scale_cheby,
                    base_activation=base_activation,
                    use_bias=use_bias,
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        for layer in self.layers:
            x = layer(x)
        return x

    def regularization_loss(self, regularize_coeffs: float = 1.0) -> float:

        return sum(
            layer.regularization_loss(regularize_coeffs)
            for layer in self.layers
        )