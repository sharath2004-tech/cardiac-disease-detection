"""
MASTER TRAINING PIPELINE - RUN EVERYTHING
==========================================

This script runs the complete pipeline:
1. Create balanced dataset (SMOTE-Tomek)
2. Train advanced model (Focal Loss + Label Smoothing + Mixup)
3. Optionally train ensemble
4. Evaluate with TTA

Expected Final Accuracy: 90-95%
"""

import subprocess
import sys
import logging
from pathlib import Path
from datetime import datetime
import json


class PipelineConfig:
    """Configuration for the complete pipeline"""
    
    # What to run
    run_create_balanced_dataset = True  # Step 1: Create SMOTE-Tomek dataset
    run_advanced_training = True         # Step 2: Train with all improvements
    run_ensemble_training = False        # Step 3: Train ensemble (takes longer)
    run_tta_evaluation = True            # Step 4: Evaluate with TTA
    
    # Paths
    base_dir = Path(__file__).parent
    output_dir = base_dir / "experiments" / "complete_pipeline"
    
    # Logging
    log_level = logging.INFO


def setup_logging(config):
    """Setup logging for the pipeline"""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = config.output_dir / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=config.log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8', errors='replace'),
            logging.StreamHandler()
        ]
    )
    
    return log_file


def run_script(script_name, description, base_dir):
    """Run a Python script and capture output"""
    
    logging.info("="*80)
    logging.info(f"RUNNING: {description}")
    logging.info(f"Script: {script_name}")
    logging.info("="*80)
    
    script_path = base_dir / script_name
    
    if not script_path.exists():
        logging.error(f"Script not found: {script_path}")
        return False
    
    try:
        # Run the script
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(base_dir),
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',  # Replace invalid characters instead of failing
            timeout=7200  # 2 hours timeout
        )
        
        # Log output (handle encoding errors)
        if result.stdout:
            logging.info("Output:")
            for line in result.stdout.split('\n'):
                if line.strip():
                    try:
                        logging.info(f"  {line}")
                    except UnicodeEncodeError:
                        # Skip lines with problematic characters
                        logging.info("  [output line with special characters]")
        
        if result.stderr:
            logging.warning("Errors/Warnings:")
            for line in result.stderr.split('\n'):
                if line.strip():
                    try:
                        logging.warning(f"  {line}")
                    except UnicodeEncodeError:
                        logging.warning("  [error line with special characters]")
        
        if result.returncode == 0:
            logging.info(f"SUCCESS: {description} completed!")
            return True
        else:
            logging.error(f"FAILED: {description} - return code {result.returncode}")
            return False
            
    except subprocess.TimeoutExpired:
        logging.error(f"TIMEOUT: {description} - exceeded 2 hours")
        return False
    except Exception as e:
        logging.error(f"ERROR: {description} - {str(e)}")
        return False


def main():
    config = PipelineConfig()
    
    # Setup logging
    log_file = setup_logging(config)
    
    logging.info("="*80)
    logging.info("COMPLETE TRAINING PIPELINE")
    logging.info("="*80)
    logging.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"Base directory: {config.base_dir}")
    logging.info(f"Output directory: {config.output_dir}")
    logging.info(f"Log file: {log_file}")
    logging.info("")
    logging.info("Pipeline steps:")
    logging.info(f"  1. Create balanced dataset: {'YES' if config.run_create_balanced_dataset else 'SKIP'}")
    logging.info(f"  2. Advanced training: {'YES' if config.run_advanced_training else 'SKIP'}")
    logging.info(f"  3. Ensemble training: {'YES' if config.run_ensemble_training else 'SKIP'}")
    logging.info(f"  4. TTA evaluation: {'YES' if config.run_tta_evaluation else 'SKIP'}")
    logging.info("="*80)
    
    results = {}
    overall_success = True
    
    # Step 1: Create balanced dataset
    if config.run_create_balanced_dataset:
        success = run_script(
            "create_balanced_dataset.py",
            "Step 1: Create Balanced Dataset (SMOTE-Tomek)",
            config.base_dir
        )
        results['balanced_dataset'] = success
        overall_success = overall_success and success
        
        if not success:
            logging.error("Pipeline stopped due to failure in dataset creation")
            return
    
    # Step 2: Advanced training
    if config.run_advanced_training:
        success = run_script(
            "train_advanced.py",
            "Step 2: Advanced Training (Focal Loss + Label Smoothing + Mixup)",
            config.base_dir
        )
        results['advanced_training'] = success
        overall_success = overall_success and success
        
        if not success:
            logging.warning("Advanced training failed, but continuing pipeline...")
    
    # Step 3: Ensemble training (optional)
    if config.run_ensemble_training:
        success = run_script(
            "train_ensemble.py",
            "Step 3: Ensemble Training (5 models)",
            config.base_dir
        )
        results['ensemble_training'] = success
        overall_success = overall_success and success
        
        if not success:
            logging.warning("Ensemble training failed, but continuing pipeline...")
    
    # Step 4: TTA evaluation
    if config.run_tta_evaluation:
        success = run_script(
            "evaluate_tta_advanced.py",
            "Step 4: TTA Evaluation",
            config.base_dir
        )
        results['tta_evaluation'] = success
        overall_success = overall_success and success
    
    # Summary
    logging.info("\n" + "="*80)
    logging.info("PIPELINE SUMMARY")
    logging.info("="*80)
    logging.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info("")
    logging.info("Results:")
    for step, success in results.items():
        status = "SUCCESS" if success else "FAILED"
        logging.info(f"  {step}: {status}")
    
    logging.info("")
    if overall_success:
        logging.info("PIPELINE COMPLETED SUCCESSFULLY!")
        logging.info("")
        logging.info("Next steps:")
        logging.info("  1. Check experiments/advanced_training/best_model.pth for the trained model")
        logging.info("  2. Review experiments/tta_results/tta_comparison.json for TTA results")
        if config.run_ensemble_training:
            logging.info("  3. Check experiments/ensemble/ for ensemble models")
        logging.info("")
        logging.info("Expected accuracy: 90-95% (with all improvements)")
    else:
        logging.error("PIPELINE COMPLETED WITH ERRORS")
        logging.info("Check the log file for details: " + str(log_file))
    
    logging.info("="*80)
    
    # Save summary
    summary = {
        'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'results': {k: str(v) for k, v in results.items()},
        'overall_success': overall_success,
        'log_file': str(log_file)
    }
    
    with open(config.output_dir / 'pipeline_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.warning("\nPipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logging.error(f"\nPipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
