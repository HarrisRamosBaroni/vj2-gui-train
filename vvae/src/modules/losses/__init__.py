# Lazy imports - only needed for training
try:
    from src.modules.losses.contperceptual import (
        LPIPSWithDiscriminator,
        MSEWithDiscriminator,
        LPIPSWithDiscriminator3D,
    )
except ImportError as e:
    # If imports fail (e.g., missing 'taming'), create placeholder
    # These are only needed for training, not inference
    import warnings
    warnings.warn(f"Could not import losses (only needed for training): {e}")
    LPIPSWithDiscriminator = None
    MSEWithDiscriminator = None
    LPIPSWithDiscriminator3D = None
