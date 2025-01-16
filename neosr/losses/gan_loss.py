from torch import Tensor, nn
from neosr.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class gan_loss(nn.Module):
    """Define GAN loss with dynamic weight scheduling.

    Args:
    ----
        gan_type (str): Support 'bce', 'mse' (l2), 'huber'.
        real_label_val (float): The value for real label. Default: 1.0.
        fake_label_val (float): The value for fake label. Default: 0.0.
        loss_weight (float): Initial loss weight. Default: 0.1.
        schedule_type (str): Type of weight scheduling ('step', 'linear', 'exponential').
            Default: None.
        schedule_steps (list[int], optional): Iteration milestones for step scheduling.
        schedule_values (list[float], optional): Corresponding loss weights at each step.
        linear_target (float, optional): Target loss weight for linear scheduling.
        linear_steps (int, optional): Total iterations for linear scheduling.
        exp_target (float, optional): Target loss weight for exponential scheduling.
        exp_growth_rate (float, optional): Growth rate for exponential scheduling.

    """

    def __init__(
        self,
        gan_type: str = "bce",
        real_label_val: float = 1.0,
        fake_label_val: float = 0.0,
        loss_weight: float = 0.1,
        schedule_type: str | None = None,
        schedule_steps: list[int] | None = None,
        schedule_values: list[float] | None = None,
        linear_target: float = 0.3,
        linear_steps: int = 20000,
        exp_target: float = 0.3,
        exp_growth_rate: float = 1.01,
    ) -> None:
        super().__init__()
        self.gan_type = gan_type
        self.loss_weight = loss_weight
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val
        self.loss: nn.BCEWithLogitsLoss | nn.MSELoss | nn.HuberLoss

        if self.gan_type == "bce":
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == "mse":
            self.loss = nn.MSELoss()
        elif self.gan_type == "huber":
            self.loss = nn.HuberLoss()
        else:
            msg = f"GAN type {self.gan_type} is not implemented."
            raise NotImplementedError(msg)

        # Schedule parameters
        self.schedule_type = schedule_type
        self.schedule_steps = schedule_steps or []
        self.schedule_values = schedule_values or []
        self.linear_target = linear_target
        self.linear_steps = linear_steps
        self.exp_target = exp_target
        self.exp_growth_rate = exp_growth_rate

        if self.schedule_type == "step" and len(self.schedule_steps) != len(self.schedule_values):
            raise ValueError("schedule_steps and schedule_values must have the same length.")

    def update_loss_weight(self, current_iter: int) -> None:
        """Update loss weight based on current iteration and schedule type.

        Args:
        ----
            current_iter (int): Current training iteration.

        """
        if self.schedule_type == "step":
            for step, value in zip(self.schedule_steps, self.schedule_values):
                if current_iter >= step:
                    self.loss_weight = value
                else:
                    break

        elif self.schedule_type == "linear":
            if current_iter <= self.linear_steps:
                self.loss_weight = (
                    self.loss_weight
                    + (self.linear_target - self.loss_weight) * (current_iter / self.linear_steps)
                )
            else:
                self.loss_weight = self.linear_target

        elif self.schedule_type == "exponential":
            if current_iter <= self.linear_steps:
                self.loss_weight = self.loss_weight * (self.exp_growth_rate ** current_iter)
                self.loss_weight = min(self.loss_weight, self.exp_target)

    def get_target_label(self, net_output: Tensor, target_is_real: bool) -> Tensor:
        """Get target label.

        Args:
        ----
            net_output (Tensor): Input tensor.
            target_is_real (bool): Whether the target is real or fake.

        Returns:
        -------
            (bool | Tensor): Target tensor. Return bool for wgan, otherwise,
                return Tensor.

        """
        target_val = self.real_label_val if target_is_real else self.fake_label_val
        return net_output.new_ones(net_output.size()) * target_val

    def forward(
        self, net_output: Tensor, target_is_real: bool, is_disc: bool = False, current_iter: int = 0
    ) -> Tensor:
        """Args:
        ----
            net_output (Tensor): The input for the loss module, i.e., the network
                prediction.
            target_is_real (bool): Whether the targe is real or fake.
            is_disc (bool): Whether the loss is for discriminators or not.
                Default: False.
            current_iter (int): Current training iteration to update loss weight.

        Returns
        -------
            Tensor: GAN loss value.

        """
        # Update loss weight based on the current iteration
        if not is_disc:  # Only update for generator loss
            self.update_loss_weight(current_iter)

        target_label = self.get_target_label(net_output, target_is_real)
        loss = self.loss(net_output, target_label)

        # loss_weight is always 1.0 for discriminators
        return loss if is_disc else loss * self.loss_weight

