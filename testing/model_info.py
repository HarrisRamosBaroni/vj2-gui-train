import torch
import torch.nn as nn

def get_model_info(model):
    """
    Simple function to get comprehensive model information.
    
    Args:
        model: PyTorch model
        
    Returns:
        dict: Model statistics including parameters, size, and layer details
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Calculate model size in bytes
    model_size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    model_size_mb = model_size_bytes / (1024 ** 2)
    model_size_gb = model_size_mb / 1024
    
    # Get layer details
    layers = []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Only leaf modules
            layer_params = sum(p.numel() for p in module.parameters())
            layer_size_mb = sum(p.numel() * p.element_size() for p in module.parameters()) / (1024 ** 2)
            
            layers.append({
                'name': name,
                'type': type(module).__name__,
                'parameters': layer_params,
                'size_mb': layer_size_mb,
                'has_params': layer_params > 0
            })
    
    return {
        'model_name': type(model).__name__,
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params,
        'model_size_mb': model_size_mb,
        'model_size_gb': model_size_gb,
        'layers': layers,
        'num_layers': len(layers)
    }

def print_model_info(model_info):
    """
    Print model information in a readable format.
    
    Args:
        model_info: Dictionary returned by get_model_info()
    """
    print("=" * 60)
    print(f"MODEL: {model_info['model_name']}")
    print("=" * 60)
    
    print(f"Total Parameters:      {model_info['total_parameters']:,}")
    print(f"Trainable Parameters:  {model_info['trainable_parameters']:,}")
    print(f"Non-trainable Params:  {model_info['non_trainable_parameters']:,}")
    print(f"Model Size:            {model_info['model_size_mb']:.2f} MB")
    print(f"Model Size:            {model_info['model_size_gb']:.4f} GB")
    print(f"Number of Layers:      {model_info['num_layers']}")
    
    print("\nLAYER BREAKDOWN:")
    print("-" * 80)
    print(f"{'Layer Name':<40} {'Type':<15} {'Params':<12} {'Size(MB)'}")
    print("-" * 80)
    
    # Show layers with parameters
    for layer in model_info['layers']:
        if layer['has_params']:
            print(f"{layer['name']:<40} {layer['type']:<15} {layer['parameters']:<12,} {layer['size_mb']:.3f}")
    
    print("\nACTIVATION FUNCTIONS & OTHER LAYERS:")
    print("-" * 80)
    print(f"{'Layer Name':<40} {'Type':<15}")
    print("-" * 80)
    
    # Show layers without parameters (activations, etc.)
    for layer in model_info['layers']:
        if not layer['has_params']:
            print(f"{layer['name']:<40} {layer['type']:<15}")

# How to use it
def analyze_my_model(model, verbose=True):
    """
    Simple usage function - just pass your model.
    
    Args:
        model: Your PyTorch model instance
    """
    info = get_model_info(model)
    if verbose:
        print_model_info(info)
    return info


if __name__ == "__main__":
    # Your existing code...
    from vj2gui_predictor import VJ2GUIPredictor
    predictor = VJ2GUIPredictor(num_frames=16, depth = 24)
    
    # Analyze the model
    model_stats = analyze_my_model(predictor, verbose=False)
    print(model_stats)
