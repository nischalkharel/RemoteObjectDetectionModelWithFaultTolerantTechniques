"""
Aggregate Fault Injection Results

Reads result JSONs from NVBitFI fault injection campaign and generates summary statistics.
Compares faulty predictions against golden (fault-free) predictions to classify outcomes.

Usage:
  python aggregate_fault_results.py \
    --technique notechnique \
    --results-dir output/notechnique_fault/results \
    --golden-dir output/notechnique_nofault/golden_predictions \
    --output output/notechnique_fault/summary_report.json \
    --iou-threshold 0.8 \
    --ap-tolerance 0.05
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
from collections import Counter
import statistics as stats

# Import AP calculation and fault classification functions
from compare_results import calculate_ap_for_image, classify_fault_outcome


def load_golden_prediction(golden_dir, image_name):
    """
    Load golden prediction for a given image.

    Args:
        golden_dir (Path): Directory containing golden predictions
        image_name (str): Base name of image (e.g., "P0019.png")

    Returns:
        list: Golden predictions in format [{'label_id': int, 'box_xyxyn': [...], 'score': float}, ...]
              Returns empty list if file exists but has no detections (valid case)
              Returns None if file doesn't exist (error case)
    """
    golden_dir = Path(golden_dir)
    image_stem = Path(image_name).stem
    golden_file = golden_dir / f"{image_stem}.txt"

    if not golden_file.exists():
        return None

    golden_preds = []
    try:
        with open(golden_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    # YOLO format: class_id xc yc w h
                    class_id = int(parts[0])
                    yolo_coords = [float(x) for x in parts[1:5]]

                    # Convert YOLO format to xyxyn format for consistency
                    xc, yc, w, h = yolo_coords
                    x1 = xc - w / 2
                    y1 = yc - h / 2
                    x2 = xc + w / 2
                    y2 = yc + h / 2

                    golden_preds.append({
                        'label_id': class_id,
                        'box_xyxyn': [x1, y1, x2, y2],
                        'score': 1.0  # Golden predictions don't have scores
                    })
    except Exception as e:
        print(f"  Warning: Error reading golden prediction {golden_file}: {e}")
        return None

    return golden_preds


def aggregate_results(results_dir, golden_dir, iou_threshold, ap_tolerance):
    """
    Aggregate all result JSONs and classify fault outcomes.

    Args:
        results_dir (Path): Directory containing result JSON files
        golden_dir (Path): Directory containing golden prediction files
        iou_threshold (float): IoU threshold for AP calculation
        ap_tolerance (float): Tolerance for MASKED classification

    Returns:
        dict: Statistics dictionary with aggregated results
    """
    results_dir = Path(results_dir)
    golden_dir = Path(golden_dir)

    # Find all result JSON files
    result_files = sorted(results_dir.glob("result_*.json"))
    total_trials = len(result_files)

    print(f"Found {total_trials} result files in {results_dir}")

    if total_trials == 0:
        print("Warning: No result files found!")
        return {
            'total_trials': 0,
            'outcomes': {},
            'metrics': {}
        }

    # Counters for outcomes
    outcomes = Counter()
    ap_values = []
    ap_values_masked = []
    inference_times = []
    error_count = 0

    # Process each result file
    for idx, result_file in enumerate(result_files, 1):
        if idx % 100 == 0:
            print(f"  Processing result {idx}/{total_trials}...")

        try:
            with open(result_file, 'r') as f:
                result = json.load(f)
        except Exception as e:
            print(f"  Error reading {result_file}: {e}")
            outcomes['CRASH'] += 1
            error_count += 1
            continue

        # Check for errors/crashes
        if result.get('error'):
            outcomes['CRASH'] += 1
            error_count += 1
            continue

        # Extract image information
        image_path = result.get('image_path')
        if not image_path:
            outcomes['CRASH'] += 1
            error_count += 1
            continue

        image_name = Path(image_path).name
        label_path = result.get('label_path')

        # Load golden prediction
        golden_preds = load_golden_prediction(golden_dir, image_name)

        if golden_preds is None:
            print(f"  Warning: No golden prediction found for {image_name}")
            outcomes['ERROR'] += 1
            error_count += 1
            continue

        # Get faulty predictions
        faulty_preds = result.get('predictions', [])

        # Classify fault outcome
        outcome = classify_fault_outcome(
            golden_predictions=golden_preds,
            faulty_predictions=faulty_preds,
            status="SUCCESS"
        )

        outcomes[outcome] += 1

        # Calculate AP if label path exists
        if label_path and Path(label_path).exists():
            ap = calculate_ap_for_image(faulty_preds, label_path)
            if ap >= 0:  # Valid AP calculation
                ap_values.append(ap)
                if outcome == "MASKED":
                    ap_values_masked.append(ap)

        # Record inference time
        inference_time = result.get('inference_time_ms', 0)
        if inference_time > 0:
            inference_times.append(inference_time)

    # Calculate statistics
    print("\nCalculating statistics...")

    # Outcome counts
    masked_count = outcomes.get('MASKED', 0)
    crash_count = outcomes.get('CRASH', 0) + outcomes.get('ERROR', 0)

    # SDC count: any outcome containing "SDC"
    sdc_count = sum(count for outcome, count in outcomes.items()
                    if 'SDC' in outcome and outcome != 'MASKED')

    # Calculate rates
    masked_rate = (masked_count / total_trials) if total_trials > 0 else 0
    sdc_rate = (sdc_count / total_trials) if total_trials > 0 else 0
    crash_rate = (crash_count / total_trials) if total_trials > 0 else 0

    # Calculate average metrics
    avg_ap_all = stats.mean(ap_values) if ap_values else 0.0
    avg_ap_masked = stats.mean(ap_values_masked) if ap_values_masked else 0.0
    avg_inference_time = stats.mean(inference_times) if inference_times else 0.0

    # Prepare statistics dictionary
    statistics_dict = {
        'total_trials': total_trials,
        'outcomes': dict(outcomes),
        'metrics': {
            'average_ap_all': avg_ap_all,
            'average_ap_masked': avg_ap_masked,
            'average_inference_time_ms': avg_inference_time,
            'masked_count': masked_count,
            'masked_rate': masked_rate,
            'sdc_count': sdc_count,
            'sdc_rate': sdc_rate,
            'crash_count': crash_count,
            'crash_rate': crash_rate,
            'valid_ap_calculations': len(ap_values),
            'valid_inference_times': len(inference_times)
        }
    }

    return statistics_dict


def main():
    """Main entry point for fault result aggregation."""
    parser = argparse.ArgumentParser(
        description="Aggregate Fault Injection Results"
    )

    parser.add_argument('--technique', type=str, required=True,
                       help='Technique name (notechnique, tmr, etc.)')
    parser.add_argument('--results-dir', type=str, required=True,
                       help='Directory containing result JSON files')
    parser.add_argument('--golden-dir', type=str, required=True,
                       help='Directory containing golden prediction files')
    parser.add_argument('--output', type=str, required=True,
                       help='Output summary JSON file path')
    parser.add_argument('--iou-threshold', type=float, default=0.8,
                       help='IoU threshold for AP calculation (default: 0.8)')
    parser.add_argument('--ap-tolerance', type=float, default=0.05,
                       help='AP tolerance for MASKED classification (default: 0.05)')

    args = parser.parse_args()

    print("=" * 60)
    print("FAULT INJECTION RESULT AGGREGATION")
    print("=" * 60)
    print(f"Technique: {args.technique}")
    print(f"Results Directory: {args.results_dir}")
    print(f"Golden Directory: {args.golden_dir}")
    print(f"Output: {args.output}")
    print(f"IoU Threshold: {args.iou_threshold}")
    print(f"AP Tolerance: {args.ap_tolerance}")
    print("=" * 60)

    # Aggregate results
    statistics_dict = aggregate_results(
        args.results_dir,
        args.golden_dir,
        args.iou_threshold,
        args.ap_tolerance
    )

    # Create final report
    report = {
        'technique': args.technique,
        'timestamp': datetime.now().isoformat(),
        'configuration': {
            'iou_threshold': args.iou_threshold,
            'ap_tolerance': args.ap_tolerance,
            'results_dir': args.results_dir,
            'golden_dir': args.golden_dir
        },
        **statistics_dict
    }

    # Save report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total Trials: {statistics_dict['total_trials']}")
    print(f"MASKED: {statistics_dict['metrics']['masked_count']} "
          f"({statistics_dict['metrics']['masked_rate']*100:.2f}%)")
    print(f"SDC: {statistics_dict['metrics']['sdc_count']} "
          f"({statistics_dict['metrics']['sdc_rate']*100:.2f}%)")
    print(f"CRASH: {statistics_dict['metrics']['crash_count']} "
          f"({statistics_dict['metrics']['crash_rate']*100:.2f}%)")

    print(f"\nAverage AP (all trials): {statistics_dict['metrics']['average_ap_all']:.4f}")
    print(f"Average AP (masked only): {statistics_dict['metrics']['average_ap_masked']:.4f}")
    print(f"Average Inference Time: {statistics_dict['metrics']['average_inference_time_ms']:.2f} ms")

    print(f"\nDetailed Outcome Breakdown:")
    for outcome, count in sorted(statistics_dict['outcomes'].items()):
        percentage = (count / statistics_dict['total_trials'] * 100) if statistics_dict['total_trials'] > 0 else 0
        print(f"  {outcome}: {count} ({percentage:.2f}%)")

    print("=" * 60)
    print(f"Report saved to: {output_path}")
    print("=" * 60)


if __name__ == '__main__':
    main()
