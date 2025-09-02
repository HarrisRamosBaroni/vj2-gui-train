import torch
import os

def load_model(
    model_path: str,
    device: torch.device,
    model_registry: dict,
    model_type: str = None,
    prepare_for_inference: bool = True
) -> torch.nn.Module:
    """
    Loads a model from a checkpoint file.

    This function automatically instantiates a model from the configuration
    stored in the checkpoint and loads the trained weights.

    Args:
        model_path (str): Path to the .pt checkpoint file.
        device (torch.device): The device to load the model onto (e.g., torch.device("cuda")).
        model_registry (dict): A dictionary mapping model type strings to model classes.
        model_type (str, optional): The type of the model to load. If omitted, the function
                                    will infer it from the `model_type` field within the
                                    checkpoint's `predictor_config`.
        prepare_for_inference (bool): If True, sets the model to evaluation mode
                                      (`model.eval()`) and disables gradients
                                      (`model.requires_grad_(False)`). Set this to
                                      False when loading a model to resume training.

    Returns:
        torch.nn.Module: The loaded model.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Checkpoint file not found at {model_path}")

    print(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)

    # --- 1. Infer Model Type if not provided ---
    if model_type is None:
        if "model_type" in checkpoint.get("predictor_config", {}):
            model_type = checkpoint["predictor_config"]["model_type"]
        else:
            # Attempt to infer from filename as a last resort
            for key in model_registry.keys():
                if key in os.path.basename(model_path):
                    model_type = key
                    break
        if model_type is None:
            raise ValueError(
                "Could not determine model_type. Please specify it explicitly."
            )
    
    # --- 2. Instantiate Model from Config ---
    if "predictor_config" not in checkpoint:
        raise ValueError(
            "Checkpoint is missing 'predictor_config'. "
            "Cannot instantiate the model automatically."
        )

    config = checkpoint["predictor_config"]
    model_class = model_registry[model_type]

    # Filter config to only pass relevant args to the model constructor
    model_arg_names = model_class.__init__.__code__.co_varnames
    model_args = {k: v for k, v in config.items() if k in model_arg_names}

    # Handle string-to-class conversion for norm_layer
    if "norm_layer" in model_args and isinstance(model_args["norm_layer"], str):
        if model_args["norm_layer"] == "nn.LayerNorm":
            model_args["norm_layer"] = torch.nn.LayerNorm
        else:
            raise ValueError(f"Unknown norm_layer string: {model_args['norm_layer']}")
            
    model = model_class(**model_args).to(device)

    # --- 3. Load State Dictionary ---
    if "predictor" not in checkpoint:
        raise ValueError("Checkpoint is missing 'predictor' state dictionary.")
        
    state_dict = checkpoint["predictor"]

    # Remove `module.` prefix if present (from DDP training)
    new_state_dict = {
        k.replace("module.", ""): v for k, v in state_dict.items()
    }

    model.load_state_dict(new_state_dict)

    # --- 4. Prepare for Inference (Optional) ---
    if prepare_for_inference:
        model.eval()
        model.requires_grad_(False)

    print(f"✅ Model '{model_type}' loaded successfully.")
    return model