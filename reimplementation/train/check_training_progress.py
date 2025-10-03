#!/usr/bin/env python3
"""
Training Progress Checker
Analyzes training logs to extract and display loss trends and metrics
"""

import re
import sys
from pathlib import Path
from typing import List, Dict, Tuple
import argparse

def extract_epoch_losses(log_file: Path) -> List[Tuple[int, float, float]]:
    """Extract epoch number, loss, and time from log file."""
    epoch_data = []

    if not log_file.exists():
        print(f"Log file not found: {log_file}")
        return epoch_data

    with open(log_file, 'r') as f:
        content = f.read()

    # Pattern: Epoch X completed in Y.Zs | Average Loss: W.VVVV
    pattern = r'Epoch (\d+) completed in ([\d.]+)s \| Average Loss: ([\d.]+)'
    matches = re.findall(pattern, content)

    for match in matches:
        epoch = int(match[0])
        time_sec = float(match[1])
        loss = float(match[2])
        epoch_data.append((epoch, loss, time_sec))

    return epoch_data

def extract_validation_losses(log_file: Path) -> List[Tuple[str, float]]:
    """Extract validation losses and metrics."""
    val_data = []

    if not log_file.exists():
        return val_data

    with open(log_file, 'r') as f:
        content = f.read()

    # Find all validation result blocks
    val_blocks = re.findall(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*?Validation Results.*?\n(.*?(?=\[\d{4}-\d{2}-\d{2}|\Z))', content, re.DOTALL)

    for timestamp, block in val_blocks:
        # Extract total loss
        total_loss_match = re.search(r'Total Loss: ([\d.]+)', block)
        if total_loss_match:
            val_data.append((timestamp, float(total_loss_match.group(1))))

    return val_data

def extract_nds_map(log_file: Path) -> List[Tuple[str, float, float]]:
    """Extract NDS and mAP metrics."""
    metrics_data = []

    if not log_file.exists():
        return metrics_data

    with open(log_file, 'r') as f:
        lines = f.readlines()

    timestamp = None
    nds = None
    map_score = None

    for line in lines:
        # Look for timestamp
        time_match = re.match(r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]', line)
        if time_match:
            timestamp = time_match.group(1)

        # Look for NDS
        if 'NDS:' in line and timestamp:
            nds_match = re.search(r'NDS: ([\d.]+)', line)
            if nds_match:
                nds = float(nds_match.group(1))

        # Look for mAP
        if 'mAP:' in line and timestamp and nds is not None:
            map_match = re.search(r'mAP: ([\d.]+)', line)
            if map_match:
                map_score = float(map_match.group(1))
                metrics_data.append((timestamp, nds, map_score))
                nds = None
                map_score = None

    return metrics_data

def format_duration(seconds: float) -> str:
    """Format duration in seconds to human readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.0f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"

def print_progress_report(log_file: Path):
    """Print comprehensive training progress report."""
    print("=" * 80)
    print("BEVFORMER TRAINING PROGRESS REPORT")
    print("=" * 80)
    print(f"Log file: {log_file}")
    print()

    # Extract data
    epoch_data = extract_epoch_losses(log_file)
    val_data = extract_validation_losses(log_file)
    metrics_data = extract_nds_map(log_file)

    if not epoch_data:
        print("❌ No epoch completion data found in log file")
        return

    # Training Loss Analysis
    print("📈 TRAINING LOSS PROGRESS")
    print("-" * 40)

    for i, (epoch, loss, time_sec) in enumerate(epoch_data):
        if i == 0:
            improvement = "baseline"
        else:
            prev_loss = epoch_data[i-1][1]
            pct_change = ((loss - prev_loss) / prev_loss) * 100
            improvement = f"{pct_change:+.1f}%"

        duration = format_duration(time_sec)
        print(f"Epoch {epoch:2d}: {loss:.4f} ({improvement:>8s}) | Time: {duration}")

    # Overall improvement
    if len(epoch_data) > 1:
        start_loss = epoch_data[0][1]
        end_loss = epoch_data[-1][1]
        total_improvement = ((end_loss - start_loss) / start_loss) * 100
        print(f"\n🎯 Overall Improvement: {total_improvement:+.1f}% ({start_loss:.4f} → {end_loss:.4f})")

    # Validation Loss Analysis
    if val_data:
        print(f"\n📊 VALIDATION LOSS PROGRESS")
        print("-" * 40)

        for i, (timestamp, val_loss) in enumerate(val_data):
            if i == 0:
                improvement = "baseline"
            else:
                prev_loss = val_data[i-1][1]
                pct_change = ((val_loss - prev_loss) / prev_loss) * 100
                improvement = f"{pct_change:+.1f}%"

            # Extract just time from timestamp
            time_only = timestamp.split()[1]
            print(f"Val {i:2d}: {val_loss:.4f} ({improvement:>8s}) | Time: {time_only}")

    # Metrics Analysis
    if metrics_data:
        print(f"\n🎯 VALIDATION METRICS (NDS/mAP)")
        print("-" * 40)

        for i, (timestamp, nds, map_score) in enumerate(metrics_data):
            time_only = timestamp.split()[1]
            print(f"Eval {i:2d}: NDS={nds:.4f}, mAP={map_score:.4f} | Time: {time_only}")

    # Training Speed Analysis
    if len(epoch_data) > 1:
        avg_time = sum(time_sec for _, _, time_sec in epoch_data) / len(epoch_data)
        print(f"\n⚡ TRAINING SPEED")
        print("-" * 40)
        print(f"Average epoch time: {format_duration(avg_time)}")

        if len(epoch_data) > 0:
            total_epochs_planned = 24  # From config
            remaining_epochs = total_epochs_planned - (epoch_data[-1][0] + 1)
            eta_seconds = remaining_epochs * avg_time
            print(f"Estimated remaining: {format_duration(eta_seconds)} ({remaining_epochs} epochs)")

    print("\n" + "=" * 80)

def main():
    parser = argparse.ArgumentParser(description='Check BEVFormer training progress from logs')
    parser.add_argument('--exp-name', type=str,
                       help='Experiment name (will look in logs/{exp_name}/{exp_name}.log)')
    parser.add_argument('--log-file', type=str,
                       help='Direct path to log file')

    args = parser.parse_args()

    if args.log_file:
        log_file = Path(args.log_file)
    elif args.exp_name:
        log_file = Path(f"logs/{args.exp_name}/{args.exp_name}.log")
    else:
        # Auto-detect most recent log
        logs_dir = Path("logs")
        if not logs_dir.exists():
            print("❌ No logs directory found. Run from training directory or specify --exp-name")
            sys.exit(1)

        # Find most recent experiment
        exp_dirs = [d for d in logs_dir.iterdir() if d.is_dir()]
        if not exp_dirs:
            print("❌ No experiment directories found in logs/")
            sys.exit(1)

        # Sort by modification time, get most recent
        latest_exp = sorted(exp_dirs, key=lambda x: x.stat().st_mtime)[-1]
        log_file = latest_exp / f"{latest_exp.name}.log"
        print(f"🔍 Auto-detected experiment: {latest_exp.name}")

    print_progress_report(log_file)

if __name__ == "__main__":
    main()