from dataclasses import dataclass

import cusrl


@dataclass
class G1FlatPPOTrainerCfg(cusrl.Trainer.Factory):
    max_iterations = 10_000
    save_interval = 500
    experiment_name = "g1_tracking"
    agent_factory = cusrl.ActorCritic.Factory(
        num_steps_per_update=24,
        actor_factory=cusrl.Actor.Factory(
            backbone_factory=cusrl.Mlp.Factory(
                hidden_dims=[512, 256, 128], activation_fn="ELU", ends_with_activation=True
            ),
            distribution_factory=cusrl.NormalDist.Factory(bijector=None),
        ),
        critic_factory=cusrl.Value.Factory(
            backbone_factory=cusrl.Mlp.Factory(
                hidden_dims=[512, 256, 128], activation_fn="ELU", ends_with_activation=True
            ),
        ),
        optimizer_factory=cusrl.OptimizerFactory("Adam", defaults={"lr": 1.0e-3}),
        sampler=cusrl.AutoMiniBatchSampler(num_epochs=5, num_mini_batches=4, shuffle=False),
        hooks=[
            cusrl.hook.ObservationNormalization(),
            cusrl.hook.ValueComputation(),
            cusrl.hook.GeneralizedAdvantageEstimation(gamma=0.99, lamda=0.95),
            cusrl.hook.AdvantageNormalization(),
            cusrl.hook.ValueLoss(weight=1.0, loss_clip=0.2),
            cusrl.hook.OnPolicyPreparation(),
            cusrl.hook.PpoSurrogateLoss(),
            cusrl.hook.EntropyLoss(weight=0.005),
            cusrl.hook.GradientClipping(max_grad_norm=1.0),
            cusrl.hook.OnPolicyStatistics(sampler=cusrl.AutoMiniBatchSampler()),
            cusrl.hook.MiniBatchWiseLRSchedule(desired_kl_divergence=0.01),
        ],
    )
