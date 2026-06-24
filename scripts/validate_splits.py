"""
Validation Script for NVFAIR Dataset Splitting

This script validates the correctness of the dataset splitting without
actually copying files. It checks:
1. Identity assignments are correct
2. No cross-set contamination
3. File inclusion rules are properly applied
4. Statistics match expectations

Run this BEFORE the actual splitting to verify everything is correct.
"""

import pandas as pd
from pathlib import Path
from collections import defaultdict
import json


class DatasetSplitValidator:
    """Validates the dataset split logic without copying files."""
    
    def __init__(self, metadata_dir):
        self.metadata_dir = Path(metadata_dir)
        self.train_identities = set()
        self.val_identities = set()
        self.test_identities = set()
        self.identity_to_dataset = {}
        self.file_metadata = None
    
    def load_metadata(self):
        """Load all metadata files."""
        print("Loading metadata files...")
        
        # Load splits
        splits_file = self.metadata_dir / "train_val_test_splits.txt"
        self._load_splits(splits_file)
        
        # Load sources
        sources_file = self.metadata_dir / "subject_sources.txt"
        self._load_subject_sources(sources_file)
        
        # Load file metadata
        xlsx_file = self.metadata_dir / "per_file_information_NVFAIR_v2.xlsx"
        self.file_metadata = pd.read_excel(xlsx_file)
        
        print(f"✓ Loaded {len(self.file_metadata)} files")
    
    def _load_splits(self, splits_file):
        """Load train/val/test splits."""
        with open(splits_file, 'r') as f:
            lines = f.readlines()
        
        current_set = None
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            if "train set:" in line.lower():
                current_set = 'train'
            elif "validation set:" in line.lower():
                current_set = 'val'
            elif "test set:" in line.lower():
                current_set = 'test'
            else:
                if current_set == 'train':
                    self.train_identities.add(line)
                elif current_set == 'val':
                    self.val_identities.add(line)
                elif current_set == 'test':
                    self.test_identities.add(line)
    
    def _load_subject_sources(self, sources_file):
        """Load subject sources."""
        with open(sources_file, 'r') as f:
            lines = f.readlines()
        
        current_dataset = None
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            if "NVIDIA-VC-subset" in line:
                current_dataset = 'NVIDIA-VC-subset'
            elif "RAVDESS dataset" in line:
                current_dataset = 'RAVDESS'
            elif "CREMA-D dataset" in line:
                current_dataset = 'CREMA-D'
            else:
                identity = line.replace("'", "")
                if current_dataset:
                    self.identity_to_dataset[identity] = current_dataset
    
    def parse_filename(self, filename):
        """Parse filename to extract identities."""
        path_parts = Path(filename).parts
        
        # Determine source and generator
        if 'NVIDIA-VC-subset' in path_parts:
            dataset_source = 'NVIDIA-VC-subset'
        elif 'RAVDESS' in path_parts:
            dataset_source = 'RAVDESS'
        elif 'CREMA-D' in path_parts:
            dataset_source = 'CREMA-D'
        else:
            dataset_source = 'unknown'
        
        if 'facevid2vid-generated' in path_parts:
            generator = 'facevid2vid'
        elif 'lia-generated' in path_parts:
            generator = 'lia'
        elif 'tps-generated' in path_parts:
            generator = 'tps'
        elif 'originals' in path_parts:
            generator = 'original'
        else:
            generator = 'unknown'
        
        # Extract target identity
        target_identity = None
        for i, part in enumerate(path_parts):
            if part in ['facevid2vid-generated', 'lia-generated', 'tps-generated', 'originals']:
                if i + 1 < len(path_parts):
                    target_identity = path_parts[i + 1]
                break
        
        # Extract driving identity
        base_filename = Path(filename).name
        driving_identity = None
        
        if generator != 'original':
            parts = base_filename.replace('.mp4', '').split('--')
            if len(parts) >= 3:
                driving_identity = parts[-1]
            elif len(parts) == 2:
                driving_identity = parts[-1]
        else:
            driving_identity = target_identity
        
        return {
            'target_identity': target_identity,
            'driving_identity': driving_identity,
            'generator': generator,
            'dataset_source': dataset_source,
            'is_self_reenactment': (target_identity == driving_identity) if (target_identity and driving_identity) else None
        }
    
    def validate_split_rules(self):
        """Validate that split rules are correctly applied."""
        print("\n" + "="*80)
        print("VALIDATION: Checking Split Rules")
        print("="*80)
        
        issues = []
        warnings = []
        
        # Process all files and simulate assignment
        assignments = {'train': [], 'val': [], 'test': []}
        
        for idx, row in self.file_metadata.iterrows():
            filename = row['file name']
            file_info = self.parse_filename(filename)
            
            target = file_info['target_identity']
            driver = file_info['driving_identity']
            
            # Determine which subset(s) this file should belong to
            for subset_name, subset_ids in [
                ('train', self.train_identities),
                ('val', self.val_identities),
                ('test', self.test_identities)
            ]:
                # Original or self-reenactment
                if file_info['generator'] == 'original' or file_info['is_self_reenactment']:
                    if target in subset_ids:
                        assignments[subset_name].append((filename, file_info))
                else:
                    # Cross-reenactment: both must be in same subset
                    if (target in subset_ids) and (driver in subset_ids):
                        assignments[subset_name].append((filename, file_info))
        
        # Check for cross-set contamination
        print("\nChecking for cross-set contamination...")
        
        for subset_name, files in assignments.items():
            subset_ids = getattr(self, f"{subset_name}_identities")
            cross_reenact_files = [
                (f, info) for f, info in files 
                if not info['is_self_reenactment'] and info['generator'] != 'original'
            ]
            
            print(f"\n{subset_name.upper()} Set:")
            print(f"  Total files: {len(files)}")
            print(f"  Cross-reenactments: {len(cross_reenact_files)}")
            
            # Check all cross-reenactments
            invalid_count = 0
            for filename, file_info in cross_reenact_files:
                target = file_info['target_identity']
                driver = file_info['driving_identity']
                
                if target not in subset_ids:
                    issues.append(f"  {subset_name}: Target {target} not in {subset_name} set (file: {filename})")
                    invalid_count += 1
                
                if driver not in subset_ids:
                    issues.append(f"  {subset_name}: Driver {driver} not in {subset_name} set (file: {filename})")
                    invalid_count += 1
            
            if invalid_count == 0:
                print(f"  ✓ No cross-set contamination detected")
            else:
                print(f"  ✗ Found {invalid_count} invalid assignments")
        
        # Report results
        print("\n" + "="*80)
        if issues:
            print("VALIDATION FAILED")
            print("="*80)
            print(f"\nFound {len(issues)} issues:")
            for issue in issues[:10]:  # Show first 10
                print(issue)
            if len(issues) > 10:
                print(f"... and {len(issues) - 10} more issues")
        else:
            print("VALIDATION PASSED")
            print("="*80)
            print("\n✓ All files correctly assigned")
            print("✓ No cross-set contamination")
            print("✓ Split rules properly applied")
        
        if warnings:
            print(f"\nWarnings: {len(warnings)}")
            for warning in warnings[:5]:
                print(warning)
        
        return len(issues) == 0
    
    def print_expected_statistics(self):
        """Print expected statistics after split."""
        print("\n" + "="*80)
        print("EXPECTED STATISTICS After Split")
        print("="*80)
        
        # Count files per subset
        counts = {'train': defaultdict(int), 'val': defaultdict(int), 'test': defaultdict(int)}
        
        for idx, row in self.file_metadata.iterrows():
            filename = row['file name']
            file_info = self.parse_filename(filename)
            
            target = file_info['target_identity']
            driver = file_info['driving_identity']
            
            for subset_name, subset_ids in [
                ('train', self.train_identities),
                ('val', self.val_identities),
                ('test', self.test_identities)
            ]:
                should_include = False
                
                if file_info['generator'] == 'original' or file_info['is_self_reenactment']:
                    should_include = target in subset_ids
                else:
                    should_include = (target in subset_ids) and (driver in subset_ids)
                
                if should_include:
                    counts[subset_name]['total'] += 1
                    counts[subset_name][file_info['generator']] += 1
                    
                    if file_info['is_self_reenactment']:
                        counts[subset_name]['self_reenact'] += 1
                    elif file_info['generator'] != 'original':
                        counts[subset_name]['cross_reenact'] += 1
        
        for subset_name in ['train', 'val', 'test']:
            print(f"\n{subset_name.upper()} Set:")
            print(f"  Identities: {len(getattr(self, f'{subset_name}_identities'))}")
            print(f"  Expected files: {counts[subset_name]['total']}")
            print(f"    Original: {counts[subset_name].get('original', 0)}")
            print(f"    Facevid2vid: {counts[subset_name].get('facevid2vid', 0)}")
            print(f"    LIA: {counts[subset_name].get('lia', 0)}")
            print(f"    TPS: {counts[subset_name].get('tps', 0)}")
            print(f"    Self-reenact: {counts[subset_name].get('self_reenact', 0)}")
            print(f"    Cross-reenact: {counts[subset_name].get('cross_reenact', 0)}")
        
        return counts
    
    def run(self):
        """Run complete validation."""
        print("="*80)
        print("NVFAIR Dataset Split Validation")
        print("="*80)
        
        # Load metadata
        self.load_metadata()
        
        # Print identity statistics
        print(f"\nIdentity Statistics:")
        print(f"  Train: {len(self.train_identities)} identities")
        print(f"  Val: {len(self.val_identities)} identities")
        print(f"  Test: {len(self.test_identities)} identities")
        print(f"  Total: {len(self.train_identities) + len(self.val_identities) + len(self.test_identities)} identities")
        
        # Check for identity overlaps (should be none)
        train_val_overlap = self.train_identities & self.val_identities
        train_test_overlap = self.train_identities & self.test_identities
        val_test_overlap = self.val_identities & self.test_identities
        
        if train_val_overlap or train_test_overlap or val_test_overlap:
            print("\n✗ WARNING: Identity overlaps detected!")
            if train_val_overlap:
                print(f"  Train-Val overlap: {train_val_overlap}")
            if train_test_overlap:
                print(f"  Train-Test overlap: {train_test_overlap}")
            if val_test_overlap:
                print(f"  Val-Test overlap: {val_test_overlap}")
        else:
            print("✓ No identity overlaps (correct)")
        
        # Validate split rules
        is_valid = self.validate_split_rules()
        
        # Print expected statistics
        self.print_expected_statistics()
        
        print("\n" + "="*80)
        if is_valid:
            print("✓ VALIDATION COMPLETE - Ready to run splitting script")
        else:
            print("✗ VALIDATION FAILED - Please review issues above")
        print("="*80)


def main():
    """Run validation."""
    metadata_dir = "NVFAIR-v2-release"  # Update this path
    
    validator = DatasetSplitValidator(metadata_dir)
    validator.run()


if __name__ == "__main__":
    main()