"""
YOLO training script for NEU-DET industrial defect detection.

Author: MaxML154
Created: 2026-07-27
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import yaml

try:
    from ultralytics import YOLO
    import torch
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("Install dependencies: pip install -r requirements.txt")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).parent))
from data.indus_argumentation import get_augmentation_config, get_training_hyperparams


def load_model_config(config_path: Path) -> dict:
    """Load model-specific configuration from YAML file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Model config not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8-sig') as f:
        config = yaml.safe_load(f)

    return config


def setup_training_dir(project: str = 'outputs', name: str = None) -> Path:
    """Create timestamped training output directory."""
    if name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"train_{timestamp}"

    output_dir = Path(project) / name
    return output_dir


def print_system_info():
    """Print system and environment information."""
    import ultralytics
    print("=" * 70)
    print("System Information")
    print("=" * 70)
    print(f"Ultralytics:      {ultralytics.__version__}")
    print(f"PyTorch version:  {torch.__version__}")
    print(f"CUDA available:   {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version:     {torch.version.cuda}")
        print(f"GPU device:       {torch.cuda.get_device_name(0)}")
        print(f"GPU memory:       {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print("=" * 70)
    print()


def print_training_config(args, train_params):
    """Print training configuration summary."""
    print("=" * 70)
    print("Training Configuration")
    print("=" * 70)
    print(f"Dataset config:   {args.data}")
    print(f"Model:            {args.model}")
    print(f"Epochs:           {train_params.get('epochs', args.epochs)}")
    print(f"Image size:       {train_params.get('imgsz', 224)}")
    print(f"Batch size:       {train_params.get('batch', 'auto')}")
    print(f"Optimizer:        {train_params.get('optimizer', 'AdamW')}")
    print(f"Learning rate:    {train_params.get('lr0', 0.001)}")
    print(f"AMP enabled:      {train_params.get('amp', True)}")
    print(f"Augmentation:     {args.aug_mode}")
    print(f"Device:           {args.device}")
    print(f"Workers:          {train_params.get('workers', 4)}")
    print("=" * 70)
    print()


def resolve_model_path(model_name: str) -> str:
    """
    Resolve model path with priority: absolute path > models/ dir > Ultralytics cache.

    Args:
        model_name: Model filename (e.g., 'yolo11s.pt') or path

    Returns:
        Resolved model path as string
    """
    model_path = Path(model_name)

    # If absolute path or file exists at given path, use directly
    if model_path.is_absolute():
        if model_path.exists():
            print(f"✓ Using model at: {model_path}")
            return str(model_path)
        else:
            print(f"✗ Model not found at: {model_path}")
            return str(model_name)

    # Check if relative path exists
    if model_path.exists():
        print(f"✓ Using existing model: {model_path}")
        return str(model_path)

    # Check project models/ directory first
    project_model = Path('models') / model_name
    if project_model.exists():
        print(f"✓ Found model in models/ directory")
        return str(project_model)

    # Check Ultralytics cache locations
    cache_locations = [
        Path.home() / '.cache' / 'ultralytics' / model_name,
        Path.home() / '.config' / 'Ultralytics' / model_name,
    ]

    for loc in cache_locations:
        if loc.exists():
            print(f"✓ Found cached model at: {loc}")
            return str(loc)

    # Not found - will be downloaded by Ultralytics
    print(f"Model '{model_name}' not found locally")
    print(f"Ultralytics will auto-download from: https://github.com/ultralytics/assets/releases")
    print(f"Downloaded models will be cached by Ultralytics")

    return str(model_name)


def train(args):
    """Execute YOLO training pipeline."""

    if args.verbose:
        print_system_info()

    # Load model config if provided
    model_config = {}
    if args.config:
        config_path = Path(args.config)
        model_config = load_model_config(config_path)
        print(f"✓ Loaded model config: {config_path}")

        # Override args with config values if not explicitly provided
        if 'model' in model_config and args.model == 'yolo11s.pt':
            args.model = model_config['model']
        if 'epochs' in model_config and args.epochs is None:
            args.epochs = model_config['epochs']
        if 'imgsz' in model_config and args.imgsz is None:
            args.imgsz = model_config['imgsz']

    # Validate data config
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {data_path}")

    # Extract model size variant (n/s/m/l/x)
    model_name = Path(args.model).stem.lower()
    if 'n' in model_name and model_name.endswith('n'):
        model_size = 'n'
    elif 's' in model_name and model_name.endswith('s'):
        model_size = 's'
    elif 'm' in model_name and model_name.endswith('m'):
        model_size = 'm'
    elif 'l' in model_name and model_name.endswith('l'):
        model_size = 'l'
    elif 'x' in model_name and model_name.endswith('x'):
        model_size = 'x'
    else:
        model_size = 's'

    # Get augmentation config and training hyperparameters
    aug_config = get_augmentation_config(args.aug_mode)
    train_params = get_training_hyperparams(model_size)

    # Merge configs
    train_params.update(aug_config)
    if model_config:
        train_params.update(model_config.get('train_params', {}))

    # CLI overrides
    if args.epochs is not None:
        train_params['epochs'] = args.epochs
    if args.imgsz is not None:
        train_params['imgsz'] = args.imgsz
    if args.batch is not None:
        train_params['batch'] = args.batch
    if args.device is not None:
        train_params['device'] = args.device
    if args.workers is not None:
        train_params['workers'] = args.workers
    if args.patience is not None:
        train_params['patience'] = args.patience
    if args.lr0 is not None:
        train_params['lr0'] = args.lr0
    if args.resume:
        train_params['resume'] = True

    # Setup output directory
    output_dir = setup_training_dir(project=args.project, name=args.name)
    train_params['project'] = str(output_dir.parent)
    train_params['name'] = output_dir.name

    print_training_config(args, train_params)

    # Confirm training
    final_epochs = train_params.get('epochs', 100)
    if final_epochs > 50 and not args.yes:
        response = input(f"Start training for {final_epochs} epochs? (y/N): ")
        if response.lower() not in ['y', 'yes']:
            print("Training cancelled.")
            return

    # Resolve model path
    print(f"\nResolving model: {args.model}")
    model_path = resolve_model_path(args.model)

    try:
        model = YOLO(model_path)
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"\n✗ Failed to load model: {e}")
        print("\nPossible solutions:")
        print("  1. Check internet connection (required for auto-download)")
        print("  2. Manually download from: https://github.com/ultralytics/assets/releases")
        print(f"  3. Place model in models/ directory: models/{Path(args.model).name}")
        return False

    print("\nStarting training...")
    print("=" * 70)

    try:
        results = model.train(
            data=str(data_path),
            **train_params
        )

        print("\n" + "=" * 70)
        print("✓ Training completed successfully!")

        actual_save_dir = results.save_dir if hasattr(results, 'save_dir') else None
        if actual_save_dir:
            print(f"Output directory: {actual_save_dir}")
            print(f"Best weights:     {Path(actual_save_dir) / 'weights' / 'best.pt'}")
            print(f"Last weights:     {Path(actual_save_dir) / 'weights' / 'last.pt'}")
        else:
            print(f"Output directory: {output_dir}")
            print(f"Best weights:     {output_dir / 'weights' / 'best.pt'}")
            print(f"Last weights:     {output_dir / 'weights' / 'last.pt'}")

        if hasattr(results, 'results_dict'):
            metrics = results.results_dict
            print("\nFinal Metrics:")
            print(f"  mAP50:     {metrics.get('metrics/mAP50(B)', 'N/A')}")
            print(f"  mAP50-95:  {metrics.get('metrics/mAP50-95(B)', 'N/A')}")

        print("=" * 70)

        return True

    except KeyboardInterrupt:
        print("\n✗ Training interrupted by user")
        return False
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Train YOLO models on NEU-DET industrial defect detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train YOLOv11s for 100 epochs
    python train_yolo.py --data configs/neudet.yaml --model yolo11s.pt

    # Train with model config
    python train_yolo.py --config configs/yolo11s.yaml

    # Resume from checkpoint
    python train_yolo.py --model runs/detect/train_xxx/weights/last.pt --resume

Augmentation Modes:
    default  - Balanced augmentation (recommended)
    strong   - Aggressive augmentation for small datasets
    enhanced - Advanced techniques for mAP > 0.80
    edge     - Minimal augmentation for edge deployment
        """
    )

    parser.add_argument(
        '--data',
        type=str,
        default='configs/neudet.yaml',
        help='Path to dataset YAML config'
    )

    parser.add_argument(
        '--model',
        type=str,
        default='yolo11s.pt',
        help='Model architecture (e.g., yolo11s.pt, yolo26n.pt)'
    )

    parser.add_argument(
        '--config',
        type=str,
        help='Path to model config YAML (overrides other params)'
    )

    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--imgsz', type=int, help='Input image size')
    parser.add_argument('--batch', type=int, help='Batch size')
    parser.add_argument('--device', type=str, default='', help='Device: "", "cpu", "0", "0,1", etc.')
    parser.add_argument('--workers', type=int, help='Number of dataloader workers')
    parser.add_argument('--patience', type=int, help='Early stopping patience')
    parser.add_argument('--lr0', type=float, help='Initial learning rate')

    parser.add_argument(
        '--aug-mode',
        type=str,
        default='default',
        choices=['default', 'strong', 'enhanced', 'edge'],
        help='Augmentation preset'
    )

    parser.add_argument('--project', type=str, default='runs/detect', help='Output project directory')
    parser.add_argument('--name', type=str, help='Experiment name')
    parser.add_argument('--resume', action='store_true', help='Resume from last checkpoint')
    parser.add_argument('--yes', '-y', action='store_true', help='Skip confirmation prompt')
    parser.add_argument('--verbose', '-v', action='store_true', help='Print system information')

    args = parser.parse_args()

    success = train(args)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
