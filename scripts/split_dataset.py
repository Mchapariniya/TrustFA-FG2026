"""
NVFAIR Dataset Splitting Script

This script splits the NVFAIR dataset into train, validation, and test subsets
according to the paper "Avatar Fingerprinting for Authorized Use of Synthetic Talking-Head Videos"
and the provided metadata files.

Key principles from the paper:
1. Splits are identity-based (not file-based)
2. NO cross-set cross-reenactments: training identities can only drive other training identities
3. Each subset includes:
   - Original videos for identities in that subset
   - Self-reenactments where target = driving identity (both in subset)
   - Cross-reenactments where BOTH target AND driving identities are in the same subset

Author: Based on NVFAIR dataset documentation and paper
"""

import os
import shutil
import pandas as pd
import re
from pathlib import Path
from collections import defaultdict
import json

class NVFAIRDatasetSplitter:
    """
    Splits NVFAIR dataset into train/val/test according to paper specifications.
    """
    
    def __init__(self, dataset_root, output_root, metadata_dir):
        """
        Initialize the splitter.
        
        Args:
            dataset_root: Root directory containing NVFAIR dataset
            output_root: Root directory for output train/val/test folders
            metadata_dir: Directory containing metadata files
        """
        self.dataset_root = Path(dataset_root)
        self.output_root = Path(output_root)
        self.metadata_dir = Path(metadata_dir)
        
        # Create output directories
        self.train_dir = self.output_root / "train"
        self.val_dir = self.output_root / "val"
        self.test_dir = self.output_root / "test"
        
        for dir_path in [self.train_dir, self.val_dir, self.test_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Data structures to store metadata
        self.train_identities = set()
        self.val_identities = set()
        self.test_identities = set()
        
        self.identity_to_dataset = {}  # Maps identity to source dataset (NVIDIA-VC/RAVDESS/CREMA-D)
        self.file_metadata = None  # Will store per-file information
        
        # Statistics
        self.stats = {
            'train': defaultdict(int),
            'val': defaultdict(int),
            'test': defaultdict(int)
        }
        
        print(f"Initialized NVFAIR Dataset Splitter")
        print(f"Dataset root: {self.dataset_root}")
        print(f"Output root: {self.output_root}")
        print(f"Metadata directory: {self.metadata_dir}")
    
    def load_metadata(self):
        """
        Load all metadata files:
        1. train_val_test_splits.txt - identity assignments
        2. subject_sources.txt - which dataset each identity belongs to
        3. per_file_information_NVFAIR_v2.xlsx - detailed file metadata
        """
        print("\n" + "="*80)
        print("STEP 1: Loading Metadata Files")
        print("="*80)
        
        # 1. Load train/val/test splits
        print("\n1.1 Loading train/val/test splits...")
        splits_file = self.metadata_dir / "train_val_test_splits.txt"
        self._load_splits(splits_file)
        
        # 2. Load subject sources
        print("\n1.2 Loading subject sources...")
        sources_file = self.metadata_dir / "subject_sources.txt"
        self._load_subject_sources(sources_file)
        
        # 3. Load per-file information
        print("\n1.3 Loading per-file information...")
        xlsx_file = self.metadata_dir / "per_file_information_NVFAIR_v2.xlsx"
        self._load_file_metadata(xlsx_file)
        
        print("\n✓ Metadata loading complete")
    
    def _load_splits(self, splits_file):
        """Load train/val/test identity splits from text file."""
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
                # This is an identity
                if current_set == 'train':
                    self.train_identities.add(line)
                elif current_set == 'val':
                    self.val_identities.add(line)
                elif current_set == 'test':
                    self.test_identities.add(line)
        
        print(f"   Train identities: {len(self.train_identities)}")
        print(f"   Val identities: {len(self.val_identities)}")
        print(f"   Test identities: {len(self.test_identities)}")
        print(f"   Total identities: {len(self.train_identities) + len(self.val_identities) + len(self.test_identities)}")
    
    def _load_subject_sources(self, sources_file):
        """Load which dataset each identity belongs to."""
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
                # This is an identity (remove any trailing apostrophes like "1085'")
                identity = line.replace("'", "")
                if current_dataset:
                    self.identity_to_dataset[identity] = current_dataset
        
        print(f"   NVIDIA-VC-subset identities: {sum(1 for d in self.identity_to_dataset.values() if d == 'NVIDIA-VC-subset')}")
        print(f"   RAVDESS identities: {sum(1 for d in self.identity_to_dataset.values() if d == 'RAVDESS')}")
        print(f"   CREMA-D identities: {sum(1 for d in self.identity_to_dataset.values() if d == 'CREMA-D')}")
    
    def _load_file_metadata(self, xlsx_file):
        """Load detailed per-file metadata from Excel."""
        self.file_metadata = pd.read_excel(xlsx_file)
        print(f"   Loaded metadata for {len(self.file_metadata)} files")
        print(f"   Columns: {self.file_metadata.columns.tolist()}")
    
    def parse_filename(self, filename):
        """
        Parse synthetic video filename to extract target and driving identities.
        
        Filename pattern: {target_identity}--{driving_video_basename}--{driving_identity}.mp4
        Example: id001--1019_IEO_FEA_LO--1019.mp4
        
        Returns:
            dict with 'target_identity', 'driving_identity', 'generator', 'dataset_source'
        """
        # Extract the path components
        path_parts = Path(filename).parts
        
        # Determine source dataset and generator
        if 'NVIDIA-VC-subset' in path_parts:
            dataset_source = 'NVIDIA-VC-subset'
        elif 'RAVDESS' in path_parts:
            dataset_source = 'RAVDESS'
        elif 'CREMA-D' in path_parts:
            dataset_source = 'CREMA-D'
        else:
            dataset_source = 'unknown'
        
        # Determine generator
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
        
        # Extract target identity (the folder name containing the file)
        # For synthetic: NVIDIA-VC-subset/facevid2vid-generated/id001/...
        # For originals: NVIDIA-VC-subset/originals/id001/...
        target_identity = None
        for i, part in enumerate(path_parts):
            if part in ['facevid2vid-generated', 'lia-generated', 'tps-generated', 'originals']:
                if i + 1 < len(path_parts):
                    target_identity = path_parts[i + 1]
                break
        
        # Extract driving identity from filename
        # Pattern: target--driving_video--driving_identity.mp4
        base_filename = Path(filename).name
        driving_identity = None
        
        if generator != 'original':
            # For synthetic videos, parse the filename
            # Example: id001--1019_IEO_FEA_LO--1019.mp4
            parts = base_filename.replace('.mp4', '').split('--')
            if len(parts) >= 3:
                driving_identity = parts[-1]
            elif len(parts) == 2:
                # Sometimes format is target--driving.mp4
                driving_identity = parts[-1]
        else:
            # For original videos, target = driving
            driving_identity = target_identity
        
        return {
            'target_identity': target_identity,
            'driving_identity': driving_identity,
            'generator': generator,
            'dataset_source': dataset_source,
            'is_self_reenactment': (target_identity == driving_identity) if (target_identity and driving_identity) else None
        }
    
    def should_include_file(self, file_info, subset_identities):
        """
        Determine if a file should be included in a subset based on paper's rules.
        
        Rules:
        1. For original videos: include if identity is in subset
        2. For self-reenactments: include if identity is in subset
        3. For cross-reenactments: include ONLY if BOTH target AND driving identities are in subset
           (This prevents cross-set cross-reenactments)
        
        Args:
            file_info: dict with target_identity, driving_identity, is_self_reenactment
            subset_identities: set of identities in the subset
        
        Returns:
            bool: True if file should be included
        """
        target = file_info['target_identity']
        driver = file_info['driving_identity']
        
        # Rule 1: Original videos - include if identity is in subset
        if file_info['generator'] == 'original':
            return target in subset_identities
        
        # Rule 2: Self-reenactments - include if identity is in subset
        if file_info['is_self_reenactment']:
            return target in subset_identities
        
        # Rule 3: Cross-reenactments - BOTH identities must be in same subset
        # This is the critical rule to prevent cross-set cross-reenactments
        return (target in subset_identities) and (driver in subset_identities)
    
    def copy_file_to_subset(self, source_file, file_info, subset_name):
        """
        Copy a file to the appropriate subset directory.
        
        Creates a structured directory layout:
        {subset_name}/
            {dataset_source}/
                {generator}/
                    {target_identity}/
                        filename.mp4
        """
        subset_dir = getattr(self, f"{subset_name}_dir")
        
        # Create directory structure
        target_dir = subset_dir / file_info['dataset_source'] / file_info['generator'] / file_info['target_identity']
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy file
        source_path = self.dataset_root / source_file
        target_path = target_dir / Path(source_file).name
        
        if source_path.exists():
            shutil.copy2(source_path, target_path)
            return target_path
        else:
            print(f"   WARNING: Source file not found: {source_path}")
            return None
    
    def process_files(self):
        """
        Process all files and copy them to appropriate subsets.
        """
        print("\n" + "="*80)
        print("STEP 2: Processing and Copying Files")
        print("="*80)
        
        # Create dataframes to store the mapping
        train_records = []
        val_records = []
        test_records = []
        
        total_files = len(self.file_metadata)
        processed = 0
        
        print(f"\nProcessing {total_files} files...")
        
        for idx, row in self.file_metadata.iterrows():
            filename = row['file name']
            processed += 1
            
            if processed % 10000 == 0:
                print(f"   Processed {processed}/{total_files} files ({100*processed/total_files:.1f}%)")
            
            # Parse filename to extract identities
            file_info = self.parse_filename(filename)
            
            # Determine which subset(s) this file belongs to
            for subset_name, subset_identities in [
                ('train', self.train_identities),
                ('val', self.val_identities),
                ('test', self.test_identities)
            ]:
                if self.should_include_file(file_info, subset_identities):
                    # Copy file
                    target_path = self.copy_file_to_subset(filename, file_info, subset_name)
                    
                    if target_path:
                        # Record metadata
                        record = {
                            'original_path': filename,
                            'new_path': str(target_path.relative_to(self.output_root)),
                            'target_identity': file_info['target_identity'],
                            'driving_identity': file_info['driving_identity'],
                            'generator': file_info['generator'],
                            'dataset_source': file_info['dataset_source'],
                            'is_self_reenactment': file_info['is_self_reenactment'],
                            'real_or_synthetic': row['real or synthetic?'],
                            'self_cross_label': row['self / cross-reenactment?'],
                            'license': row['applicable license'],
                            'research_use': row['file can be used for research topics other than audio-visual remote communication?'],
                            'publication_use': row['file can attached in publication-related materials?']
                        }
                        
                        if subset_name == 'train':
                            train_records.append(record)
                            self.stats['train'][file_info['generator']] += 1
                        elif subset_name == 'val':
                            val_records.append(record)
                            self.stats['val'][file_info['generator']] += 1
                        else:
                            test_records.append(record)
                            self.stats['test'][file_info['generator']] += 1
        
        print(f"\n✓ File processing complete")
        
        # Save CSV files
        print("\n" + "="*80)
        print("STEP 3: Saving Metadata CSV Files")
        print("="*80)
        
        train_df = pd.DataFrame(train_records)
        val_df = pd.DataFrame(val_records)
        test_df = pd.DataFrame(test_records)
        
        train_csv = self.output_root / "train_files.csv"
        val_csv = self.output_root / "val_files.csv"
        test_csv = self.output_root / "test_files.csv"
        
        train_df.to_csv(train_csv, index=False)
        val_df.to_csv(val_csv, index=False)
        test_df.to_csv(test_csv, index=False)
        
        print(f"   Train CSV saved: {train_csv} ({len(train_df)} files)")
        print(f"   Val CSV saved: {val_csv} ({len(val_df)} files)")
        print(f"   Test CSV saved: {test_csv} ({len(test_df)} files)")
        
        return train_df, val_df, test_df
    
    def print_statistics(self, train_df, val_df, test_df):
        """Print detailed statistics about the split."""
        print("\n" + "="*80)
        print("STEP 4: Dataset Split Statistics")
        print("="*80)
        
        for subset_name, df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
            print(f"\n{subset_name} Set:")
            print(f"  Total files: {len(df)}")
            
            # By generator
            print(f"\n  By Generator:")
            for gen, count in df['generator'].value_counts().items():
                print(f"    {gen}: {count}")
            
            # By dataset source
            print(f"\n  By Dataset Source:")
            for source, count in df['dataset_source'].value_counts().items():
                print(f"    {source}: {count}")
            
            # Self vs Cross
            print(f"\n  Self vs Cross-reenactment:")
            if 'is_self_reenactment' in df.columns:
                self_count = df['is_self_reenactment'].sum()
                cross_count = (~df['is_self_reenactment']).sum()
                orig_count = df['generator'].eq('original').sum()
                print(f"    Self-reenactment: {self_count}")
                print(f"    Cross-reenactment: {cross_count}")
                print(f"    Original: {orig_count}")
            
            # Unique identities
            print(f"\n  Unique Target Identities: {df['target_identity'].nunique()}")
            print(f"  Unique Driving Identities: {df['driving_identity'].nunique()}")
        
        # Verify no cross-set contamination
        print("\n" + "="*80)
        print("VERIFICATION: No Cross-Set Cross-Reenactments")
        print("="*80)
        
        # Check train cross-reenactments
        train_cross = train_df[train_df['is_self_reenactment'] == False]
        train_cross_drivers = set(train_cross['driving_identity'].unique())
        train_cross_targets = set(train_cross['target_identity'].unique())
        
        # These should all be in train identities
        invalid_drivers_in_train = train_cross_drivers - self.train_identities
        invalid_targets_in_train = train_cross_targets - self.train_identities
        
        print(f"\nTrain Set Cross-Reenactments:")
        print(f"  Total cross-reenactments: {len(train_cross)}")
        print(f"  Unique driving identities: {len(train_cross_drivers)}")
        print(f"  Unique target identities: {len(train_cross_targets)}")
        print(f"  Invalid drivers (not in train set): {len(invalid_drivers_in_train)}")
        print(f"  Invalid targets (not in train set): {len(invalid_targets_in_train)}")
        
        if invalid_drivers_in_train:
            print(f"    WARNING: Found invalid drivers: {invalid_drivers_in_train}")
        if invalid_targets_in_train:
            print(f"    WARNING: Found invalid targets: {invalid_targets_in_train}")
        
        # Similar checks for val and test
        for subset_name, df, id_set in [('Val', val_df, self.val_identities), 
                                         ('Test', test_df, self.test_identities)]:
            subset_cross = df[df['is_self_reenactment'] == False]
            cross_drivers = set(subset_cross['driving_identity'].unique())
            cross_targets = set(subset_cross['target_identity'].unique())
            
            invalid_drivers = cross_drivers - id_set
            invalid_targets = cross_targets - id_set
            
            print(f"\n{subset_name} Set Cross-Reenactments:")
            print(f"  Total cross-reenactments: {len(subset_cross)}")
            print(f"  Unique driving identities: {len(cross_drivers)}")
            print(f"  Unique target identities: {len(cross_targets)}")
            print(f"  Invalid drivers (not in {subset_name.lower()} set): {len(invalid_drivers)}")
            print(f"  Invalid targets (not in {subset_name.lower()} set): {len(invalid_targets)}")
            
            if invalid_drivers:
                print(f"    WARNING: Found invalid drivers: {invalid_drivers}")
            if invalid_targets:
                print(f"    WARNING: Found invalid targets: {invalid_targets}")
        
        print("\n" + "="*80)
        if not any([invalid_drivers_in_train, invalid_targets_in_train]):
            print("✓ VERIFICATION PASSED: No cross-set contamination detected!")
        else:
            print("✗ VERIFICATION FAILED: Cross-set contamination detected!")
        print("="*80)
    
    def save_summary(self, train_df, val_df, test_df):
        """Save a summary JSON file with key statistics."""
        summary = {
            'dataset': 'NVFAIR',
            'split_method': 'identity-based',
            'paper': 'Avatar Fingerprinting for Authorized Use of Synthetic Talking-Head Videos',
            'total_identities': len(self.train_identities) + len(self.val_identities) + len(self.test_identities),
            'splits': {
                'train': {
                    'num_identities': len(self.train_identities),
                    'num_files': len(train_df),
                    'identities': sorted(list(self.train_identities))
                },
                'val': {
                    'num_identities': len(self.val_identities),
                    'num_files': len(val_df),
                    'identities': sorted(list(self.val_identities))
                },
                'test': {
                    'num_identities': len(self.test_identities),
                    'num_files': len(test_df),
                    'identities': sorted(list(self.test_identities))
                }
            },
            'file_counts_by_generator': {
                'train': dict(train_df['generator'].value_counts()),
                'val': dict(val_df['generator'].value_counts()),
                'test': dict(test_df['generator'].value_counts())
            }
        }
        
        summary_file = self.output_root / "dataset_split_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✓ Summary saved: {summary_file}")
    
    def run(self):
        """Execute the complete dataset splitting pipeline."""
        print("\n" + "="*80)
        print("NVFAIR Dataset Splitting Pipeline")
        print("="*80)
        
        # Step 1: Load metadata
        self.load_metadata()
        
        # Step 2: Process and copy files
        train_df, val_df, test_df = self.process_files()
        
        # Step 3: Print statistics
        self.print_statistics(train_df, val_df, test_df)
        
        # Step 4: Save summary
        self.save_summary(train_df, val_df, test_df)
        
        print("\n" + "="*80)
        print("✓ Dataset splitting complete!")
        print("="*80)
        print(f"\nOutput location: {self.output_root}")
        print(f"  - train/ : {len(train_df)} files")
        print(f"  - val/   : {len(val_df)} files")
        print(f"  - test/  : {len(test_df)} files")
        print(f"  - train_files.csv : Training set metadata")
        print(f"  - val_files.csv   : Validation set metadata")
        print(f"  - test_files.csv  : Test set metadata")
        print(f"  - dataset_split_summary.json : Summary statistics")


def main():
    """
    Main function to run the dataset splitting.
    
    USAGE:
        python split_nvfair_dataset.py
    
    NOTE: You need to update the paths below to match your setup:
        - dataset_root: Where your NVFAIR dataset is located
        - output_root: Where you want the split dataset to be saved
        - metadata_dir: Where the metadata files are located
    """
    
    # UPDATE THESE PATHS TO MATCH YOUR SETUP
    dataset_root = "NVFAIR-v2-release"  # Root folder containing NVIDIA-VC-subset, RAVDESS, CREMA-D
    output_root = "NVFAIR_split"  # Where to save train/val/test
    metadata_dir = "NVFAIR-v2-release"  # Where metadata files are located
    
    # Create splitter and run
    splitter = NVFAIRDatasetSplitter(
        dataset_root=dataset_root,
        output_root=output_root,
        metadata_dir=metadata_dir
    )
    
    splitter.run()


if __name__ == "__main__":
    main()