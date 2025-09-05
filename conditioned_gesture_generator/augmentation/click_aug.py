"""
Click Augmentation Module

This module generates synthetic click action sequences for data augmentation.
It creates action sequences of the same format as the autoencoder/VAE training data:
- Sequence length: 250 timesteps
- Action format: [x, y, p] where x,y ∈ [0,1] and p ∈ {0,1}
"""

import numpy as np
import os
from pathlib import Path
from typing import Tuple, List
import random


# Action sequence configuration (matching AE/VAE model)
ACTION_SEQUENCE_LENGTH = 250
ACTION_DIM = 3  # [x, y, p]


def generate_click_signal(x: float, y: float, delta_t: int, 
                         sequence_length: int = ACTION_SEQUENCE_LENGTH) -> np.ndarray:
    """
    Generate a click action signal with specified position and duration.
    
    Args:
        x: X coordinate (float between 0 and 1)
        y: Y coordinate (float between 0 and 1) 
        delta_t: Duration of the click in timesteps
        sequence_length: Total length of the action sequence (default: 250)
        
    Returns:
        np.ndarray: Action sequence of shape [sequence_length, 3] with format [x, y, p]
                   where p=1 during the click period and p=0 otherwise
    """
    if not (0 <= x <= 1):
        raise ValueError(f"x must be between 0 and 1, got {x}")
    if not (0 <= y <= 1):
        raise ValueError(f"y must be between 0 and 1, got {y}")
    if not (0 <= delta_t <= sequence_length):
        raise ValueError(f"delta_t must be between 0 and {sequence_length}, got {delta_t}")
    
    # Initialize action sequence with zeros
    action_sequence = np.zeros((sequence_length, ACTION_DIM), dtype=np.float32)
    
    # Handle special case of delta_t = 0 (no action)
    if delta_t == 0:
        return action_sequence  # Return all zeros
    
    # Randomly choose start time for the click (ensuring it fits within sequence)
    max_start_time = sequence_length - delta_t
    start_time = random.randint(0, max_start_time)
    end_time = start_time + delta_t
    
    # Set click signal during the specified period
    action_sequence[start_time:end_time, 0] = x  # X coordinate
    action_sequence[start_time:end_time, 1] = y  # Y coordinate  
    action_sequence[start_time:end_time, 2] = 1  # Press state (p=1 during click)
    
    return action_sequence


def generate_random_click_pair(min_delta_t: int = 5, max_delta_t: int = 50,
                              sequence_length: int = ACTION_SEQUENCE_LENGTH) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a pair of random click signals with random positions and durations.
    
    Args:
        min_delta_t: Minimum click duration (default: 5)
        max_delta_t: Maximum click duration (default: 50)
        sequence_length: Total length of action sequences (default: 250)
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: Two action sequences representing different clicks
    """
    # Generate random parameters for first click
    x1 = random.uniform(0, 1)
    y1 = random.uniform(0, 1)
    delta_t1 = random.randint(min_delta_t, max_delta_t)
    
    # Generate random parameters for second click  
    x2 = random.uniform(0, 1)
    y2 = random.uniform(0, 1)
    delta_t2 = random.randint(min_delta_t, max_delta_t)
    
    # Generate both click signals
    click1 = generate_click_signal(x1, y1, delta_t1, sequence_length)
    click2 = generate_click_signal(x2, y2, delta_t2, sequence_length)
    
    return click1, click2


def generate_random_click_sequence(min_delta_t: int = 5, max_delta_t: int = 50,
                                  sequence_length: int = ACTION_SEQUENCE_LENGTH) -> np.ndarray:
    """
    Generate a single random click signal.
    
    Args:
        min_delta_t: Minimum click duration (default: 5)
        max_delta_t: Maximum click duration (default: 50) 
        sequence_length: Total length of action sequence (default: 250)
        
    Returns:
        np.ndarray: Single action sequence of shape [sequence_length, 3]
    """
    x = random.uniform(0, 1)
    y = random.uniform(0, 1)
    delta_t = random.randint(min_delta_t, max_delta_t)
    
    return generate_click_signal(x, y, delta_t, sequence_length)


def generate_swipe_signal(start_x: float, start_y: float, end_x: float, end_y: float, 
                         delta_t: int, sequence_length: int = ACTION_SEQUENCE_LENGTH) -> np.ndarray:
    """
    Generate a swipe action signal with specified start and end positions.
    
    Args:
        start_x: Starting X coordinate (float between 0 and 1)
        start_y: Starting Y coordinate (float between 0 and 1)
        end_x: Ending X coordinate (float between 0 and 1)
        end_y: Ending Y coordinate (float between 0 and 1)
        delta_t: Duration of the swipe in timesteps
        sequence_length: Total length of the action sequence (default: 250)
        
    Returns:
        np.ndarray: Action sequence of shape [sequence_length, 3] with format [x, y, p]
                   where p=1 during the swipe period and coordinates interpolate linearly
    """
    if not (0 <= start_x <= 1 and 0 <= start_y <= 1):
        raise ValueError(f"Start coordinates must be between 0 and 1, got ({start_x}, {start_y})")
    if not (0 <= end_x <= 1 and 0 <= end_y <= 1):
        raise ValueError(f"End coordinates must be between 0 and 1, got ({end_x}, {end_y})")
    if not (0 <= delta_t <= sequence_length):
        raise ValueError(f"delta_t must be between 0 and {sequence_length}, got {delta_t}")
    
    # Initialize action sequence with zeros
    action_sequence = np.zeros((sequence_length, ACTION_DIM), dtype=np.float32)
    
    # Handle special case of delta_t = 0 (no action)
    if delta_t == 0:
        return action_sequence  # Return all zeros
    
    # Randomly choose start time for the swipe (ensuring it fits within sequence)
    max_start_time = sequence_length - delta_t
    start_time = random.randint(0, max_start_time)
    end_time = start_time + delta_t
    
    # Generate linear interpolation between start and end points
    t_values = np.linspace(0, 1, delta_t)
    x_values = start_x + t_values * (end_x - start_x)
    y_values = start_y + t_values * (end_y - start_y)
    
    # Set swipe signal during the specified period
    action_sequence[start_time:end_time, 0] = x_values  # X coordinates (interpolated)
    action_sequence[start_time:end_time, 1] = y_values  # Y coordinates (interpolated)
    action_sequence[start_time:end_time, 2] = 1         # Press state (p=1 during swipe)
    
    return action_sequence


def generate_random_swipe_pair(min_delta_t: int = 5, max_delta_t: int = 50,
                              sequence_length: int = ACTION_SEQUENCE_LENGTH) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a pair of random swipe signals with random start/end positions and durations.
    
    Args:
        min_delta_t: Minimum swipe duration (default: 5)
        max_delta_t: Maximum swipe duration (default: 50)
        sequence_length: Total length of action sequences (default: 250)
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: Two action sequences representing different swipes
    """
    # Generate random parameters for first swipe
    start_x1, start_y1 = random.uniform(0, 1), random.uniform(0, 1)
    end_x1, end_y1 = random.uniform(0, 1), random.uniform(0, 1)
    delta_t1 = random.randint(min_delta_t, max_delta_t)
    
    # Generate random parameters for second swipe
    start_x2, start_y2 = random.uniform(0, 1), random.uniform(0, 1)
    end_x2, end_y2 = random.uniform(0, 1), random.uniform(0, 1)
    delta_t2 = random.randint(min_delta_t, max_delta_t)
    
    # Generate both swipe signals
    swipe1 = generate_swipe_signal(start_x1, start_y1, end_x1, end_y1, delta_t1, sequence_length)
    swipe2 = generate_swipe_signal(start_x2, start_y2, end_x2, end_y2, delta_t2, sequence_length)
    
    return swipe1, swipe2


def generate_mixed_action_signal(x_behavior: str, y_behavior: str, delta_t: int,
                                sequence_length: int = ACTION_SEQUENCE_LENGTH) -> np.ndarray:
    """
    Generate an action signal with independent X and Y behavior.
    
    Args:
        x_behavior: "constant" (click-like) or "changing" (swipe-like) for X coordinate
        y_behavior: "constant" (click-like) or "changing" (swipe-like) for Y coordinate  
        delta_t: Duration of the action in timesteps
        sequence_length: Total length of the action sequence (default: 250)
        
    Returns:
        np.ndarray: Action sequence of shape [sequence_length, 3] with format [x, y, p]
    """
    if x_behavior not in ["constant", "changing"]:
        raise ValueError(f"x_behavior must be 'constant' or 'changing', got '{x_behavior}'")
    if y_behavior not in ["constant", "changing"]:
        raise ValueError(f"y_behavior must be 'constant' or 'changing', got '{y_behavior}'")
    if not (0 <= delta_t <= sequence_length):
        raise ValueError(f"delta_t must be between 0 and {sequence_length}, got {delta_t}")
    
    # Generate random positions
    start_x, start_y = random.uniform(0, 1), random.uniform(0, 1)
    
    # Determine end positions based on behavior
    if x_behavior == "constant":
        end_x = start_x  # No change in X
    else:  # changing
        end_x = random.uniform(0, 1)  # Random end position
        
    if y_behavior == "constant":
        end_y = start_y  # No change in Y
    else:  # changing
        end_y = random.uniform(0, 1)  # Random end position
    
    # Use existing swipe generation (it handles constant cases when start == end)
    return generate_swipe_signal(start_x, start_y, end_x, end_y, delta_t, sequence_length)


def generate_random_action_sequence(min_delta_t: int = 5, max_delta_t: int = 50,
                                   sequence_length: int = ACTION_SEQUENCE_LENGTH,
                                   action_types: List[str] = None) -> np.ndarray:
    """
    Generate a random action sequence with configurable action types.
    
    Args:
        min_delta_t: Minimum action duration (default: 5)
        max_delta_t: Maximum action duration (default: 50)
        sequence_length: Total length of action sequence (default: 250)
        action_types: List of action types to choose from. Options:
                     ["click", "horizontal", "vertical", "diagonal"]
                     If None, defaults to all types.
        
    Returns:
        np.ndarray: Single action sequence of shape [sequence_length, 3]
    """
    if action_types is None:
        action_types = ["click", "horizontal", "vertical", "diagonal"]
    
    # Randomly select an action type
    action_type = random.choice(action_types)
    delta_t = random.randint(min_delta_t, max_delta_t)
    
    # Map action types to X/Y behaviors
    behavior_map = {
        "click": ("constant", "constant"),      # X=const, Y=const
        "horizontal": ("changing", "constant"), # X=change, Y=const  
        "vertical": ("constant", "changing"),   # X=const, Y=change
        "diagonal": ("changing", "changing")    # X=change, Y=change
    }
    
    x_behavior, y_behavior = behavior_map[action_type]
    return generate_mixed_action_signal(x_behavior, y_behavior, delta_t, sequence_length)


def generate_random_swipe_sequence(min_delta_t: int = 5, max_delta_t: int = 50,
                                  sequence_length: int = ACTION_SEQUENCE_LENGTH) -> np.ndarray:
    """
    Generate a single random swipe signal (backward compatibility).
    
    Args:
        min_delta_t: Minimum swipe duration (default: 5)
        max_delta_t: Maximum swipe duration (default: 50)
        sequence_length: Total length of action sequence (default: 250)
        
    Returns:
        np.ndarray: Single action sequence of shape [sequence_length, 3]
    """
    # For backward compatibility, generate diagonal swipes
    return generate_random_action_sequence(min_delta_t, max_delta_t, sequence_length, ["diagonal"])


def generate_augmented_dataset(output_dir: str = "augmented_action_data",
                              num_sequences: int = 1000,
                              min_delta_t: int = 5,
                              max_delta_t: int = 50,
                              sequence_length: int = ACTION_SEQUENCE_LENGTH,
                              action_type: str = "mixed",
                              sequences_per_file: int = 100) -> None:
    """
    Generate a dataset of augmented action sequences (clicks, swipes, or mixed) and save to files.
    
    Args:
        output_dir: Directory to save the generated data (default: "augmented_action_data")
        num_sequences: Number of action sequences to generate (default: 1000)
        min_delta_t: Minimum action duration (default: 5)
        max_delta_t: Maximum action duration (default: 50)
        sequence_length: Length of each action sequence (default: 250)
        action_type: Type of actions to generate - "click", "swipe", or "mixed" (default: "mixed")
        sequences_per_file: Number of sequences to pack per .npy file (default: 100)
    """
    # Validate action_type
    valid_types = ["click", "swipe", "mixed"]
    if action_type not in valid_types:
        raise ValueError(f"action_type must be one of {valid_types}, got '{action_type}'")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating {num_sequences} augmented {action_type} sequences...")
    print(f"Sequence length: {sequence_length}")
    print(f"Action duration range: {min_delta_t}-{max_delta_t} timesteps")
    print(f"Sequences per file: {sequences_per_file}")
    print(f"Output directory: {output_path.absolute()}")
    
    # Calculate number of files needed
    num_files = (num_sequences + sequences_per_file - 1) // sequences_per_file
    
    # Generate and save action sequences in batches
    click_count = 0
    swipe_count = 0
    
    for file_idx in range(num_files):
        # Determine how many sequences for this file
        start_seq = file_idx * sequences_per_file
        end_seq = min(start_seq + sequences_per_file, num_sequences)
        current_batch_size = end_seq - start_seq
        
        # Create batch array: [batch_size, sequence_length, action_dim]
        batch_sequences = np.zeros((current_batch_size, sequence_length, ACTION_DIM), dtype=np.float32)
        
        # Generate sequences for this batch
        for batch_idx in range(current_batch_size):
            seq_idx = start_seq + batch_idx
            
            # Determine action type for this sequence
            if action_type == "click":
                sequence = generate_random_click_sequence(min_delta_t, max_delta_t, sequence_length)
                click_count += 1
            elif action_type == "swipe":
                sequence = generate_random_swipe_sequence(min_delta_t, max_delta_t, sequence_length)
                swipe_count += 1
            else:  # mixed - includes click, horizontal, vertical, diagonal
                action_types = ["click", "horizontal", "vertical", "diagonal"]
                sequence = generate_random_action_sequence(min_delta_t, max_delta_t, sequence_length, action_types)
                
                # Count the action type for statistics
                active_indices = np.where(sequence[:, 2] == 1)[0]
                if len(active_indices) > 0:
                    start_x, start_y = sequence[active_indices[0], 0], sequence[active_indices[0], 1]
                    end_x, end_y = sequence[active_indices[-1], 0], sequence[active_indices[-1], 1]
                    
                    x_changed = abs(end_x - start_x) > 0.01
                    y_changed = abs(end_y - start_y) > 0.01
                    
                    if not x_changed and not y_changed:
                        click_count += 1
                    else:
                        swipe_count += 1
                else:
                    click_count += 1
            
            # Add sequence to batch
            batch_sequences[batch_idx] = sequence
        
        # Flatten batch for ActionBlockDataset compatibility
        # From [sequences_per_file, sequence_length, action_dim] to [sequences_per_file * sequence_length, action_dim]
        flattened_sequences = batch_sequences.reshape(-1, ACTION_DIM)
        
        # Save flattened data as single .npy file
        filename = f"batch_{file_idx:06d}_actions.npy"
        filepath = output_path / filename
        np.save(filepath, flattened_sequences)
        
        # Progress reporting
        print(f"Generated file {file_idx + 1}/{num_files}: {current_batch_size} sequences")
    
    print(f"✅ Successfully generated {num_sequences} action sequences in {num_files} files")
    
    # Generate summary statistics
    print("\n📊 Dataset Summary:")
    print(f"Total files: {num_files}")
    print(f"Total sequences: {num_sequences}")
    print(f"Sequences per file: {sequences_per_file}")
    print(f"File naming format: batch_XXXXXX_actions.npy")
    print(f"File shape: [sequences_per_file * {sequence_length}, {ACTION_DIM}] = [{sequences_per_file * sequence_length}, {ACTION_DIM}]")
    
    if action_type == "mixed":
        print(f"Action type breakdown:")
        print(f"  • Click-like sequences: {click_count} (~{click_count/num_sequences:.1%})")
        print(f"  • Swipe-like sequences: {swipe_count} (~{swipe_count/num_sequences:.1%})")
        print(f"  • Total action types: click, horizontal, vertical, diagonal")
    else:
        if action_type == "click":
            print(f"  • Click sequences: {click_count}")
        else:
            print(f"  • Swipe sequences: {swipe_count}")
    
    print(f"Data format: [x, y, p] where x,y ∈ [0,1] and p ∈ {{0,1}}")
    if action_type in ["swipe", "mixed"]:
        print(f"Action behavior: Independent X/Y coordinate control (constant or linear interpolation)")
    
    # File size estimation
    total_size_mb = (num_sequences * sequence_length * ACTION_DIM * 4) / (1024 * 1024)  # 4 bytes per float32
    avg_file_size_mb = total_size_mb / num_files
    print(f"Estimated total size: {total_size_mb:.1f} MB ({avg_file_size_mb:.1f} MB per file)")


def generate_augmented_click_dataset(output_dir: str = "augmented_click_data",
                                   num_sequences: int = 1000,
                                   min_delta_t: int = 5,
                                   max_delta_t: int = 50,
                                   sequence_length: int = ACTION_SEQUENCE_LENGTH) -> None:
    """
    Backward compatibility function for generating click-only datasets.
    
    Args:
        output_dir: Directory to save the generated data (default: "augmented_click_data")
        num_sequences: Number of click sequences to generate (default: 1000)
        min_delta_t: Minimum click duration (default: 5)
        max_delta_t: Maximum click duration (default: 50)
        sequence_length: Length of each action sequence (default: 250)
    """
    generate_augmented_dataset(
        output_dir=output_dir,
        num_sequences=num_sequences,
        min_delta_t=min_delta_t,
        max_delta_t=max_delta_t,
        sequence_length=sequence_length,
        action_type="click"
    )


def load_click_sequence(filepath: str) -> np.ndarray:
    """
    Load a click sequence from a saved numpy file.
    
    Args:
        filepath: Path to the .npy file
        
    Returns:
        np.ndarray: Action sequence of shape [sequence_length, 3]
    """
    return np.load(filepath)


def visualize_click_sequence(action_sequence: np.ndarray, title: str = "Click Sequence") -> None:
    """
    Simple visualization of a click action sequence.
    
    Args:
        action_sequence: Action sequence of shape [sequence_length, 3]
        title: Title for the visualization
    """
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        timesteps = np.arange(len(action_sequence))
        
        # Plot X coordinate
        axes[0].plot(timesteps, action_sequence[:, 0], 'b-', linewidth=2)
        axes[0].set_ylabel('X Coordinate')
        axes[0].set_ylim(-0.1, 1.1)
        axes[0].grid(True, alpha=0.3)
        
        # Plot Y coordinate  
        axes[1].plot(timesteps, action_sequence[:, 1], 'g-', linewidth=2)
        axes[1].set_ylabel('Y Coordinate')
        axes[1].set_ylim(-0.1, 1.1)
        axes[1].grid(True, alpha=0.3)
        
        # Plot press state
        axes[2].plot(timesteps, action_sequence[:, 2], 'r-', linewidth=2)
        axes[2].set_ylabel('Press State')
        axes[2].set_ylim(-0.1, 1.1)
        axes[2].set_xlabel('Timestep')
        axes[2].grid(True, alpha=0.3)
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("Matplotlib not available for visualization")


def test_augmentation_with_plots():
    """
    Test function that generates and displays 3 different click sequences.
    Creates a subplot showing x, y, p channels for each sequence.
    """
    try:
        import matplotlib.pyplot as plt
        
        print("🖱️  Testing Action Augmentation with Visualization")
        print("=" * 60)
        
        # Generate 3 different click sequences
        sequences = []
        descriptions = []
        
        # Sequence 1: Click (X=const, Y=const)
        seq1 = generate_mixed_action_signal("constant", "constant", delta_t=15)
        sequences.append(seq1)
        active_indices = np.where(seq1[:, 2] == 1)[0]
        x_pos, y_pos = seq1[active_indices[0], 0], seq1[active_indices[0], 1]
        descriptions.append(f"Click (15 timesteps)\nX=const, Y=const\nPosition: ({x_pos:.3f}, {y_pos:.3f})")
        
        # Sequence 2: Horizontal swipe (X=changing, Y=const)
        seq2 = generate_mixed_action_signal("changing", "constant", delta_t=25)
        sequences.append(seq2)
        active_indices = np.where(seq2[:, 2] == 1)[0]
        start_x, start_y = seq2[active_indices[0], 0], seq2[active_indices[0], 1]
        end_x, end_y = seq2[active_indices[-1], 0], seq2[active_indices[-1], 1]
        descriptions.append(f"Horizontal Swipe (25 timesteps)\nX=changing, Y=const\nFrom: ({start_x:.3f}, {start_y:.3f}) → ({end_x:.3f}, {end_y:.3f})")
        
        # Sequence 3: Vertical swipe (X=const, Y=changing)
        seq3 = generate_mixed_action_signal("constant", "changing", delta_t=20)
        sequences.append(seq3)
        active_indices = np.where(seq3[:, 2] == 1)[0]
        start_x, start_y = seq3[active_indices[0], 0], seq3[active_indices[0], 1]
        end_x, end_y = seq3[active_indices[-1], 0], seq3[active_indices[-1], 1]
        descriptions.append(f"Vertical Swipe (20 timesteps)\nX=const, Y=changing\nFrom: ({start_x:.3f}, {start_y:.3f}) → ({end_x:.3f}, {end_y:.3f})")
        
        # Create subplot figure
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        fig.suptitle('Action Augmentation Test - Click & Swipe Sequences', fontsize=16, fontweight='bold')
        
        colors = ['blue', 'green', 'red']
        channel_names = ['X Coordinate', 'Y Coordinate', 'Press State (p)']
        
        # Plot each sequence
        for seq_idx, (sequence, description) in enumerate(zip(sequences, descriptions)):
            timesteps = np.arange(len(sequence))
            
            for channel_idx in range(3):
                ax = axes[seq_idx, channel_idx]
                
                # Plot the channel
                ax.plot(timesteps, sequence[:, channel_idx], 
                       color=colors[channel_idx], linewidth=2, alpha=0.8)
                
                # Formatting
                if channel_idx < 2:  # x, y coordinates
                    ax.set_ylim(-0.1, 1.1)
                    ax.set_ylabel(f'{channel_names[channel_idx]}')
                else:  # press state
                    ax.set_ylim(-0.1, 1.1)
                    ax.set_ylabel(f'{channel_names[channel_idx]}')
                    # Highlight active period
                    active_mask = sequence[:, 2] == 1
                    if np.any(active_mask):
                        ax.fill_between(timesteps, 0, sequence[:, channel_idx], 
                                      where=active_mask, alpha=0.3, color='red')
                
                ax.grid(True, alpha=0.3)
                ax.set_xlim(0, len(sequence)-1)
                
                # Add title only to top row
                if seq_idx == 0:
                    ax.set_title(channel_names[channel_idx], fontweight='bold')
                
                # Add x-label only to bottom row
                if seq_idx == 2:
                    ax.set_xlabel('Timestep')
                
                # Add sequence description to leftmost column
                if channel_idx == 0:
                    ax.text(-0.25, 0.5, description, transform=ax.transAxes, 
                           rotation=90, ha='center', va='center', fontsize=10,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.93, left=0.12)
        
        # Print sequence statistics
        print("\n📊 Generated Sequence Statistics:")
        for i, (seq, desc) in enumerate(zip(sequences, descriptions)):
            active_timesteps = np.sum(seq[:, 2])
            active_indices = np.where(seq[:, 2] == 1)[0]
            start_time = active_indices[0] if len(active_indices) > 0 else -1
            end_time = active_indices[-1] if len(active_indices) > 0 else -1
            
            print(f"\nSequence {i+1}:")
            print(f"  {desc.replace(chr(10), ' | ')}")
            print(f"  Active period: timesteps {start_time} to {end_time}")
            print(f"  Total active timesteps: {int(active_timesteps)}")
        
        plt.show()
        print("\n✅ Visualization test completed!")
        
    except ImportError:
        print("❌ Error: matplotlib is required for visualization")
        print("Install with: pip install matplotlib")
        return False
    except Exception as e:
        print(f"❌ Error during visualization: {e}")
        return False
    
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Click Augmentation Module")
    parser.add_argument("--test", action="store_true", 
                       help="Run visualization test with 3 sample click sequences")
    
    args = parser.parse_args()
    
    if args.test:
        test_augmentation_with_plots()
    else:
        # Default behavior - run basic tests
        print("🖱️  Click Augmentation Module")
        print("=" * 50)
        
        # Test single click generation
        print("\n1. Testing single click generation:")
        click = generate_click_signal(x=0.5, y=0.3, delta_t=10)
        print(f"Generated click: shape={click.shape}")
        print(f"Click period: timesteps {np.where(click[:, 2] == 1)[0]}")
        print(f"Position during click: x={click[click[:, 2] == 1, 0][0]:.3f}, y={click[click[:, 2] == 1, 1][0]:.3f}")
        
        # Test mixed action generation  
        print("\n2. Testing mixed action generation:")
        
        # Test horizontal swipe (X=changing, Y=constant)
        horizontal = generate_mixed_action_signal("changing", "constant", delta_t=15)
        active_indices = np.where(horizontal[:, 2] == 1)[0]
        start_x, start_y = horizontal[active_indices[0], 0], horizontal[active_indices[0], 1]
        end_x, end_y = horizontal[active_indices[-1], 0], horizontal[active_indices[-1], 1]
        print(f"Horizontal swipe: X ({start_x:.3f} → {end_x:.3f}), Y constant ({start_y:.3f})")
        
        # Test vertical swipe (X=constant, Y=changing)
        vertical = generate_mixed_action_signal("constant", "changing", delta_t=12)
        active_indices = np.where(vertical[:, 2] == 1)[0]
        start_x, start_y = vertical[active_indices[0], 0], vertical[active_indices[0], 1]
        end_x, end_y = vertical[active_indices[-1], 0], vertical[active_indices[-1], 1]
        print(f"Vertical swipe: X constant ({start_x:.3f}), Y ({start_y:.3f} → {end_y:.3f})")
        
        # Test click (X=constant, Y=constant)
        click = generate_mixed_action_signal("constant", "constant", delta_t=10)
        active_indices = np.where(click[:, 2] == 1)[0]
        x_pos, y_pos = click[active_indices[0], 0], click[active_indices[0], 1]
        print(f"Click: X constant ({x_pos:.3f}), Y constant ({y_pos:.3f})")
        
        # Test random action generation
        print("\n3. Testing random mixed action generation:")
        for i in range(4):
            action = generate_random_action_sequence(min_delta_t=8, max_delta_t=15)
            active_indices = np.where(action[:, 2] == 1)[0]
            if len(active_indices) > 0:
                start_x, start_y = action[active_indices[0], 0], action[active_indices[0], 1]
                end_x, end_y = action[active_indices[-1], 0], action[active_indices[-1], 1]
                
                x_changed = abs(end_x - start_x) > 0.01
                y_changed = abs(end_y - start_y) > 0.01
                
                if not x_changed and not y_changed:
                    action_type = "Click"
                elif x_changed and not y_changed:
                    action_type = "Horizontal"
                elif not x_changed and y_changed:
                    action_type = "Vertical"
                else:
                    action_type = "Diagonal"
                    
                print(f"  Action {i+1}: {action_type} ({len(active_indices)} timesteps)")
        
        # Generate small mixed dataset for testing
        print("\n4. Generating test mixed dataset (10 sequences):")
        generate_augmented_dataset(
            output_dir="augmented_mixed_data",
            num_sequences=10,
            min_delta_t=5,
            max_delta_t=20,
            action_type="mixed"
        )
        
        print("\n✅ All tests completed successfully!")
        print("\n💡 Tip: Run with --test flag to see visualization of generated sequences")