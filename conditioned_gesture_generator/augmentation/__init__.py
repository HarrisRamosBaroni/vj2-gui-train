"""
Augmentation Module

This module provides data augmentation utilities for the action autoencoder/VAE training.
"""

from .click_aug import (
    # Click functions
    generate_click_signal,
    generate_random_click_pair, 
    generate_random_click_sequence,
    # Swipe functions
    generate_swipe_signal,
    generate_random_swipe_pair,
    generate_random_swipe_sequence,
    # Dataset generation functions
    generate_augmented_dataset,
    generate_augmented_click_dataset,  # backward compatibility
    # Utility functions
    load_click_sequence,
    visualize_click_sequence,
    test_augmentation_with_plots,
    # Constants
    ACTION_SEQUENCE_LENGTH,
    ACTION_DIM
)

__all__ = [
    # Click functions
    'generate_click_signal',
    'generate_random_click_pair',
    'generate_random_click_sequence',
    # Swipe functions
    'generate_swipe_signal',
    'generate_random_swipe_pair',
    'generate_random_swipe_sequence',
    # Dataset generation functions
    'generate_augmented_dataset',
    'generate_augmented_click_dataset',
    # Utility functions
    'load_click_sequence',
    'visualize_click_sequence',
    'test_augmentation_with_plots',
    # Constants
    'ACTION_SEQUENCE_LENGTH',
    'ACTION_DIM'
]