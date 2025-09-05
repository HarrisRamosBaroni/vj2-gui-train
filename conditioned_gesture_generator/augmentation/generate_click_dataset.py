#!/usr/bin/env python3
"""
Generate Augmented Click Dataset

Command-line script to generate large datasets of synthetic click sequences
for training the action autoencoder/VAE models.
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from conditioned_gesture_generator.augmentation.click_aug import generate_augmented_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic click action sequences for data augmentation"
    )
    
    # Dataset generation parameters
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="augmented_click_data",
        help="Directory to save generated click sequences (default: augmented_click_data)"
    )
    
    parser.add_argument(
        "--num_sequences", 
        type=int, 
        default=5000,
        help="Number of click sequences to generate (default: 5000)"
    )
    
    parser.add_argument(
        "--min_delta_t_pct", 
        type=float, 
        default=0.02,
        help="Minimum action duration as percentage of sequence length (0.0-1.0, default: 0.02 = 2%%, 0.0 = no action)"
    )
    
    parser.add_argument(
        "--max_delta_t_pct", 
        type=float, 
        default=0.20,
        help="Maximum action duration as percentage of sequence length (0.0-1.0, default: 0.20 = 20%%, 1.0 = full sequence)"
    )
    
    parser.add_argument(
        "--sequence_length", 
        type=int, 
        default=250,
        help="Length of each action sequence (default: 250, matching AE/VAE)"
    )
    
    parser.add_argument(
        "--action_type",
        type=str,
        default="mixed",
        choices=["click", "swipe", "mixed"],
        help="Type of actions to generate: 'click' (static), 'swipe' (diagonal), 'mixed' (click+horizontal+vertical+diagonal) (default: mixed)"
    )
    
    parser.add_argument(
        "--sequences_per_file",
        type=int,
        default=100,
        help="Number of action sequences to pack per .npy file (default: 100)"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Convert percentage to actual timesteps
    min_delta_t = int(args.min_delta_t_pct * args.sequence_length)
    max_delta_t = int(args.max_delta_t_pct * args.sequence_length)
    
    # Ensure valid range
    min_delta_t = max(0, min_delta_t)  # Allow 0 (no action)
    max_delta_t = min(args.sequence_length, max_delta_t)  # Allow 100% (full sequence)
    if min_delta_t > max_delta_t:
        max_delta_t = min_delta_t
    
    # Validation
    if args.num_sequences <= 0:
        print("❌ Error: num_sequences must be positive")
        return 1
        
    if args.min_delta_t_pct < 0 or args.max_delta_t_pct < 0:
        print("❌ Error: delta_t percentages must be non-negative")
        return 1
        
    if args.min_delta_t_pct > args.max_delta_t_pct:
        print("❌ Error: min_delta_t_pct must be less than or equal to max_delta_t_pct")
        return 1
        
    if args.max_delta_t_pct > 1.0:
        print("❌ Error: max_delta_t_pct cannot exceed 1.0 (100%)")
        return 1
        
    if max_delta_t > args.sequence_length:
        print(f"⚠️  Warning: max_delta_t ({max_delta_t}) adjusted to sequence_length ({args.sequence_length})")
        max_delta_t = args.sequence_length
    
    print("🖱️  Augmented Action Dataset Generator")
    print("=" * 50)
    print(f"📁 Output directory: {args.output_dir}")
    print(f"🔢 Number of sequences: {args.num_sequences:,}")
    print(f"🎯 Action type: {args.action_type}")
    print(f"⏱️  Action duration range: {min_delta_t}-{max_delta_t} timesteps ({args.min_delta_t_pct:.1%}-{args.max_delta_t_pct:.1%} of sequence)")
    print(f"📏 Sequence length: {args.sequence_length}")
    print()
    
    # Generate dataset
    try:
        generate_augmented_dataset(
            output_dir=args.output_dir,
            num_sequences=args.num_sequences,
            min_delta_t=min_delta_t,
            max_delta_t=max_delta_t,
            sequence_length=args.sequence_length,
            action_type=args.action_type,
            sequences_per_file=args.sequences_per_file
        )
        
        print("\n🎉 Dataset generation completed successfully!")
        print(f"\n💡 Usage Tips:")
        print(f"   • Use this data with ActionBlockDataset in training scripts")
        print(f"   • Files are compatible with existing data loaders")
        print(f"   • Mix with real data for enhanced training")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error during dataset generation: {e}")
        return 1


if __name__ == "__main__":
    exit(main())