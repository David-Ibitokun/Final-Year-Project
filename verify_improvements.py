"""
Verification Script for Model Improvements
Checks that all changes have been properly applied
"""

import re
from pathlib import Path

def check_file_content(file_path, search_patterns, description):
    """Check if file contains expected patterns"""
    print(f"\n{'='*70}")
    print(f"Checking: {file_path}")
    print(f"{'='*70}")
    
    if not Path(file_path).exists():
        print(f"❌ File not found: {file_path}")
        return False
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    all_passed = True
    for pattern, expected, check_name in search_patterns:
        if re.search(pattern, content, re.IGNORECASE):
            if expected:
                print(f"✅ {check_name}: FOUND")
            else:
                print(f"❌ {check_name}: FOUND (should not exist)")
                all_passed = False
        else:
            if expected:
                print(f"❌ {check_name}: NOT FOUND (required)")
                all_passed = False
            else:
                print(f"✅ {check_name}: NOT FOUND (correct)")
    
    return all_passed

def main():
    print("="*70)
    print("MODEL IMPROVEMENTS VERIFICATION")
    print("="*70)
    
    checks = []
    
    # 1. Check data_prep_and_features.ipynb
    checks.append(check_file_content(
        'data_prep_and_features.ipynb',
        [
            (r"CROPS\s*=\s*\['Maize',\s*'Cassava',\s*'Yams'\]", True, "3-crop list (no Rice)"),
            (r"CROPS\s*=\s*\[.*'Rice'.*\]", False, "Rice should not be in CROPS"),
            (r"3-Crop|Three Crop", True, "Documentation updated to 3-crop"),
            (r"4-Crop|Four Crop", False, "Old 4-crop references removed"),
        ],
        "Data Preparation Notebook"
    ))
    
    # 2. Check phase3_model_dev.ipynb
    checks.append(check_file_content(
        'phase3_model_dev.ipynb',
        [
            (r"CROPS\s*=\s*\['Maize',\s*'Cassava',\s*'Yams'\]", True, "3-crop list (no Rice)"),
            (r"gamma=3\.0", True, "Focal loss gamma=3.0"),
            (r"alpha=0\.3", True, "Focal loss alpha=0.3"),
            (r"learning_rate=0\.0002", True, "Learning rate 0.0002"),
            (r"batch_size=24", True, "Batch size 24"),
            (r"Dense\(192", True, "Fusion layer 192 neurons"),
            (r"Dropout\(0\.25\)", True, "Reduced dropout 0.25"),
            (r"1\.5\s*\*", True, "Amplified class weights"),
            (r"residual|Add\(\)", True, "Residual connections"),
        ],
        "Model Training Notebook"
    ))
    
    # 3. Check phase4_validation.ipynb
    checks.append(check_file_content(
        'phase4_validation.ipynb',
        [
            (r"Rice.*removed|excluded", True, "Rice exclusion documented"),
            (r"Hybrid.*fix|improv", True, "Hybrid improvements documented"),
            (r"3.*crop|Three crop", True, "3-crop system documented"),
        ],
        "Validation Notebook"
    ))
    
    # Summary
    print(f"\n{'='*70}")
    print("VERIFICATION SUMMARY")
    print(f"{'='*70}")
    
    if all(checks):
        print("✅ ALL CHECKS PASSED!")
        print("\nNext steps:")
        print("1. Run data_prep_and_features.ipynb (10-15 min)")
        print("2. Run phase3_model_dev.ipynb (3-4 hours CPU)")
        print("3. Run phase4_validation.ipynb (5-10 min)")
        print("\nExpected: Hybrid model 65-70% accuracy")
    else:
        print("❌ SOME CHECKS FAILED")
        print("\nPlease review the failed checks above and:")
        print("1. Ensure all notebooks have been properly edited")
        print("2. Save all changes")
        print("3. Re-run this verification script")
    
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
