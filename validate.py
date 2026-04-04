import sys
from pathlib import Path
def check_files():
    print("\n" + "=" * 80)
    print("CHECKING PROJECT STRUCTURE")
    print("=" * 80)
    required_files = [
        "src/core/config.py",
        "src/core/logger.py",
        "src/core/__init__.py",
        "src/data/dataset.py",
        "src/data/preprocess.py",
        "src/data/__init__.py",
        "src/utils/transforms.py",
        "src/utils/__init__.py",
        "src/models/model.py",
        "src/models/train.py",
        "src/models/evaluate.py",
        "src/models/inference.py",
        "src/models/__init__.py",
        "src/api/main.py",
        "src/api/routes.py",
        "src/api/schema.py",
        "src/api/__init__.py",
        "src/__init__.py",
        "train.py",
        "evaluate.py",
        "requirements.txt",
        ".env.example",
        "Dockerfile",
        "docker-compose.yml",
        "README.md",
        ".gitignore",
    ]
    missing = []
    for file in required_files:
        path = Path(file)
        if path.exists():
            print(f"✓ {file}")
        else:
            print(f"✗ {file} (MISSING)")
            missing.append(file)
    print("\n" + "-" * 80)
    print("CHECKING DIRECTORIES")
    print("-" * 80)
    required_dirs = [
        "data/train",
        "data/val",
        "model",
        "results",
    ]
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists() and path.is_dir():
            print(f"✓ {dir_path}/")
        else:
            print(f"✗ {dir_path}/ (MISSING)")
            missing.append(dir_path)
    return len(missing) == 0
def check_imports():
    print("\n" + "=" * 80)
    print("CHECKING IMPORTS")
    print("=" * 80)
    modules = [
        "src.core.config",
        "src.core.logger",
        "src.data.dataset",
        "src.data.preprocess",
        "src.utils.transforms",
        "src.models.model",
        "src.models.train",
        "src.models.evaluate",
        "src.models.inference",
        "src.api.schema",
        "src.api.routes",
        "src.api.main",
    ]
    failed = []
    for module in modules:
        try:
            __import__(module)
            print(f"✓ {module}")
        except Exception as e:
            print(f"✗ {module}: {e}")
            failed.append(module)
    return len(failed) == 0
def check_dependencies():
    print("\n" + "=" * 80)
    print("CHECKING DEPENDENCIES")
    print("=" * 80)
    dependencies = [
        "torch",
        "torchvision",
        "fastapi",
        "uvicorn",
        "pillow",
        "numpy",
        "sklearn",
        "pydantic",
    ]
    failed = []
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"✓ {dep}")
        except ImportError:
            print(f"✗ {dep} (NOT INSTALLED)")
            failed.append(dep)
    if failed:
        print(f"\n⚠ Missing dependencies: {', '.join(failed)}")
        print("Run: pip install -r requirements.txt")
        return False
    return True
def main():
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "  MULTI-CLASS IMAGE CLASSIFICATION - VALIDATION SCRIPT".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "=" * 78 + "╝")
    checks = [
        ("Project Structure", check_files),
        ("Module Imports", check_imports),
        ("Dependencies", check_dependencies),
    ]
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ Error during {name} check: {e}")
            results.append((name, False))
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    all_passed = all(result for _, result in results)
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{name:.<50} {status}")
    print("\n" + "=" * 80)
    if all_passed:
        print("\n✓ ALL CHECKS PASSED!")
        print("\nNext steps:")
        print("  1. python train.py              # Train model")
        print("  2. python evaluate.py           # Evaluate")
        print("  3. uvicorn src.api.main:app --reload   # Run API")
        print("  4. curl http://localhost:8000/health   # Test API")
        print("\nOr use Docker:")
        print("  docker-compose up --build")
        print("\nRead README.md for detailed instructions.")
        return 0
    else:
        print("\n✗ SOME CHECKS FAILED")
        print("Please fix the issues above before proceeding.")
        return 1
if __name__ == "__main__":
    sys.exit(main())
