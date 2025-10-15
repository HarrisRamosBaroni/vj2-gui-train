import wandb
from typing import Optional, Dict, Any

__all__ = ["init_wandb"]


def init_wandb(
    project_name: str,
    run_name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> "wandb.wandb_run.Run":
    """
    Initializes a new Weights & Biases run.

    Args:
        project_name (str): The name of the wandb project.
        run_name (Optional[str]): The name of the run. Defaults to a wandb-generated name.
        config (Optional[Dict[str, Any]]): A dictionary of hyperparameters and settings.
        **kwargs: Additional arguments to pass to wandb.init().

    Returns:
        A wandb Run object.
    """
    run = wandb.init(
        project=project_name,
        name=run_name,
        config=config,
        **kwargs,
    )
    return run