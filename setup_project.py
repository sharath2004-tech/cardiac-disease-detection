"""
ACRMF-Net Project Setup Script
Stage 1 - Module 1: Project Folder Structure Verification

Run this script to verify Stage 1 completion and setup the environment
"""

import sys
from pathlib import Path
from config import get_config, setup_logging, StageLogger


def verify_folder_structure():
    """Verify all required folders exist"""
    print("\n🔍 Verifying folder structure...")
    
    required_dirs = [
        "config",
        "data/loaders",
        "data/preprocessing",
        "models/encoders",
        "models/reliability",
        "models/fusion",
        "models/heads",
        "training",
        "evaluation",
        "notebooks",
        "results/checkpoints",
        "results/logs",
        "results/figures",
        "results/metrics",
    ]
    
    project_root = Path(__file__).parent
    missing_dirs = []
    
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if full_path.exists():
            print(f"  ✓ {dir_path}")
        else:
            print(f"  ✗ {dir_path} - MISSING")
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        print(f"\n❌ Missing {len(missing_dirs)} directories!")
        return False
    else:
        print(f"\n✅ All folders verified!")
        return True


def verify_config_files():
    """Verify configuration files exist"""
    print("\n🔍 Verifying configuration files...")
    
    required_files = [
        "config/__init__.py",
        "config/config.py",
        "config/requirements.txt",
        "config/logging_config.py",
    ]
    
    project_root = Path(__file__).parent
    missing_files = []
    
    for file_path in required_files:
        full_path = project_root / file_path
        if full_path.exists():
            print(f"  ✓ {file_path}")
        else:
            print(f"  ✗ {file_path} - MISSING")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n❌ Missing {len(missing_files)} files!")
        return False
    else:
        print(f"\n✅ All configuration files verified!")
        return True


def test_configuration():
    """Test configuration loading"""
    print("\n🔍 Testing configuration...")
    
    try:
        config = get_config()
        print("  ✓ Configuration loaded successfully")
        config.summary()
        return True
    except Exception as e:
        print(f"  ✗ Configuration error: {e}")
        return False


def test_logging():
    """Test logging setup"""
    print("\n🔍 Testing logging system...")
    
    try:
        # Setup logger
        logger = setup_logging(
            log_dir=Path(__file__).parent / "results" / "logs",
            log_level="INFO",
            experiment_name="stage1_verification"
        )
        
        # Test stage logger
        stage_logger = StageLogger(logger)
        stage_logger.log_stage_start(1, "Project Initialization")
        stage_logger.log_module_start(1, "Project Folder Structure")
        stage_logger.log_module_complete(1, "Project Folder Structure")
        stage_logger.log_module_start(2, "Configuration Module")
        stage_logger.log_module_complete(2, "Configuration Module")
        stage_logger.log_module_start(3, "Requirements Installation")
        stage_logger.log_module_complete(3, "Requirements Installation")
        stage_logger.log_module_start(4, "Logging Module")
        stage_logger.log_module_complete(4, "Logging Module")
        stage_logger.log_stage_complete(1, "Project Initialization")
        
        print("  ✓ Logging system working correctly")
        return True
    except Exception as e:
        print(f"  ✗ Logging error: {e}")
        return False


def check_datasets():
    """Check if datasets are available"""
    print("\n🔍 Checking datasets...")
    
    project_root = Path(__file__).parent
    
    datasets = {
        "PTB-XL (ECG)": project_root / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3",
        "CinC 2016 (PCG)": project_root / "archive",
        "Clinical CSV": project_root / "heart_disease_uci.csv",
    }
    
    found = 0
    for name, path in datasets.items():
        if path.exists():
            print(f"  ✓ {name}: {path}")
            found += 1
        else:
            print(f"  ⚠️  {name}: NOT FOUND at {path}")
    
    if found == len(datasets):
        print(f"\n✅ All datasets found!")
        return True
    else:
        print(f"\n⚠️  {len(datasets) - found} dataset(s) missing (optional for Stage 1)")
        return False


def print_next_steps():
    """Print next steps for Stage 2"""
    print("\n" + "=" * 80)
    print("🎉 STAGE 1 - PROJECT INITIALIZATION COMPLETE!")
    print("=" * 80)
    print("\n📋 Completed Modules:")
    print("  ✓ Module 1: Project Folder Structure")
    print("  ✓ Module 2: Configuration Module")
    print("  ✓ Module 3: Requirements Installation")
    print("  ✓ Module 4: Logging Module")
    print("\n📦 Next Steps:")
    print("  1. Install requirements:")
    print("     pip install -r config/requirements.txt")
    print("\n  2. Download datasets (if not already available):")
    print("     - PTB-XL: https://physionet.org/content/ptb-xl/1.0.3/")
    print("     - CinC 2016: https://physionet.org/content/challenge-2016/1.0.0/")
    print("\n  3. Proceed to Stage 2 - Dataset Preparation:")
    print("     - Module 5: Clinical Dataset Loader")
    print("     - Module 6: ECG Dataset Loader")
    print("     - Module 7: PCG Dataset Loader")
    print("     - Module 8: Dataset Split Module")
    print("\n" + "=" * 80)


def main():
    """Main setup verification"""
    print("=" * 80)
    print("ACRMF-Net Project Setup Verification")
    print("Stage 1: Project Initialization")
    print("=" * 80)
    
    results = {
        "Folder Structure": verify_folder_structure(),
        "Configuration Files": verify_config_files(),
        "Configuration Loading": test_configuration(),
        "Logging System": test_logging(),
        "Datasets": check_datasets(),
    }
    
    # Summary
    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)
    
    for check, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {check}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print_next_steps()
        return 0
    else:
        print("\n❌ Some checks failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
