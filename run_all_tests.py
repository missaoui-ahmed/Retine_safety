import subprocess
import sys
from pathlib import Path

def run_command(cmd, description, timeout=300):
    print(f"\n{'='*70}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print(f"{'='*70}")
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        if result.returncode == 0:
            print(f"✅ {description} PASSED")
            return True
        else:
            print(f"❌ {description} FAILED (exit code: {result.returncode})")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏱️ {description} TIMED OUT")
        return False
    except Exception as e:
        print(f"❌ {description} ERROR: {e}")
        return False

def main():
    """Run all tests"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║                   MedVision - Test Suite                     ║
║              Comprehensive System Verification               ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    tests = []
    
    # Test 1: Quick system test
    print("\n📋 Test 1/4: System Verification")
    result1 = run_command(
        "python quick_test.py",
        "System Verification Test",
        timeout=60
    )
    tests.append(("System Verification", result1))
    
    # Test 2: Single inference
    print("\n📋 Test 2/4: Single Image Inference")
    result2 = run_command(
        "python test_inference.py",
        "Single Image Inference Test",
        timeout=30
    )
    tests.append(("Single Inference", result2))
    
    # Test 3: Check if model exists
    print("\n📋 Test 3/4: Model Checkpoint Check")
    model_path = Path("checkpoints/best_model.pth")
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"✅ Model checkpoint found: {size_mb:.2f} MB")
        result3 = True
    else:
        print("❌ Model checkpoint not found. Please run training first.")
        result3 = False
    tests.append(("Model Checkpoint", result3))
    
    # Test 4: Check configuration
    print("\n📋 Test 4/4: Configuration Check")
    config_path = Path("config.yaml")
    if config_path.exists():
        print("✅ Configuration file found")
        with open(config_path, 'r') as f:
            content = f.read()
            if 'num_epochs: 5' in content:
                print("✅ Epochs set to 5 for quick training")
            if 'architecture: "resnet50"' in content:
                print("✅ Using ResNet-50 architecture")
            if 'image_size: 512' in content:
                print("✅ Image size: 512×512")
        result4 = True
    else:
        print("❌ Configuration file not found")
        result4 = False
    tests.append(("Configuration", result4))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, result in tests:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:30} {status}")
    
    passed = sum(1 for _, result in tests if result)
    total = len(tests)
    
    print("\n" + "="*70)
    print(f"TOTAL: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    print("="*70)
    
    if passed == total:
        print("""
╔══════════════════════════════════════════════════════════════╗
║              🎉 ALL TESTS PASSED! 🎉                         ║
║                                                               ║
║  Your MedVision system is working perfectly!                 ║
║                                                               ║
║  What you can do now:                                        ║
║  1. Run full training:                                       ║
║     python train.py --config config.yaml --dataset aptos    ║
║                                                               ║
║  2. Evaluate your model:                                     ║
║     python eval.py --config config.yaml --dataset aptos \\   ║
║       --checkpoint checkpoints/best_model.pth                ║
║                                                               ║
║  3. Start API server:                                        ║
║     uvicorn api.main:app --port 8000                         ║
║                                                               ║
║  📚 See TESTING_GUIDE.md for more details                    ║
╚══════════════════════════════════════════════════════════════╝
        """)
        return 0
    else:
        print("""
╔══════════════════════════════════════════════════════════════╗
║              ⚠️  SOME TESTS FAILED  ⚠️                       ║
║                                                               ║
║  Please check the errors above and:                          ║
║  1. Ensure all dependencies are installed                    ║
║  2. Check dataset paths in config.yaml                       ║
║  3. Verify GPU is available (CUDA)                           ║
║  4. Run: python quick_test.py for diagnostics                ║
║                                                               ║
║  📚 See TESTING_GUIDE.md for troubleshooting                 ║
╚══════════════════════════════════════════════════════════════╝
        """)
        return 1

if __name__ == "__main__":
    sys.exit(main())
