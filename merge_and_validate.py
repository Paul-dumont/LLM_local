#!/usr/bin/env python3
"""
Script to merge, convert and validate input/output files.
Creates: input_final and output_final folders with merged data.
"""

import os
import re
import shutil
from pathlib import Path
from typing import Dict, Set, Tuple
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

BASE_DIR = Path("/home/luciacev/Desktop/LLM_local/data_training")
FINAL_DIR = Path("/home/luciacev/Desktop/LLM_local/data_training")

INPUT_SOURCES = [
    BASE_DIR / "500" / "data_input",
    BASE_DIR / "1000" / "data_input_500",
    BASE_DIR / "1000" / "data_input_1000",
]

OUTPUT_SOURCES = [
    BASE_DIR / "500" / "data_output",
    BASE_DIR / "1000" / "data_output_500",
    BASE_DIR / "1000" / "data_output_1000",
]

# Create final directories
INPUT_FINAL = FINAL_DIR / "input_final"
OUTPUT_FINAL = FINAL_DIR / "output_final"
INPUT_TOTAL = INPUT_FINAL / "input_total"
OUTPUT_TOTAL = OUTPUT_FINAL / "output_total"


def setup_folders():
    """Create necessary folder structure"""
    logger.info("Setting up folder structure...")
    
    for folder in [INPUT_TOTAL, OUTPUT_TOTAL]:
        folder.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created: {folder}")


def extract_patient_id(filename: str) -> str:
    """Extract BXXX from filename (B001, B123, B1234, etc.)"""
    match = re.search(r'(B\d+)', filename)
    if match:
        return match.group(1)
    return None


def process_input_files() -> Dict[str, Path]:
    """
    Copy and convert all input files to input_total folder
    Returns: dict mapping patient_id to filepath
    """
    logger.info("Processing input files...")
    patient_files = {}
    file_count = 0
    duplicate_count = 0
    
    for source_dir in INPUT_SOURCES:
        if not source_dir.exists():
            logger.warning(f"Source not found: {source_dir}")
            continue
        
        logger.info(f"Processing: {source_dir}")
        
        for input_file in source_dir.glob("*"):
            if not input_file.is_file():
                continue
            
            patient_id = extract_patient_id(input_file.name)
            if not patient_id:
                logger.warning(f"Could not extract patient ID from: {input_file.name}")
                continue
            
            # Create output filename (remove _Word_text suffix and ensure .txt)
            output_filename = f"{patient_id}.txt"
            output_path = INPUT_TOTAL / output_filename
            
            # Check for duplicates
            if patient_id in patient_files:
                logger.warning(
                    f"Duplicate patient ID {patient_id}: "
                    f"keeping {patient_files[patient_id].name}, "
                    f"skipping {input_file.name}"
                )
                duplicate_count += 1
                continue
            
            # Copy file
            try:
                shutil.copy2(input_file, output_path)
                patient_files[patient_id] = output_path
                file_count += 1
            except Exception as e:
                logger.error(f"Error copying {input_file}: {e}")
    
    logger.info(f"✓ Processed {file_count} input files")
    logger.info(f"  Duplicates skipped: {duplicate_count}")
    return patient_files


def process_output_files() -> Dict[str, Path]:
    """
    Copy and convert all output files to output_total folder
    Returns: dict mapping patient_id to filepath
    """
    logger.info("Processing output files...")
    patient_files = {}
    file_count = 0
    duplicate_count = 0
    
    for source_dir in OUTPUT_SOURCES:
        if not source_dir.exists():
            logger.warning(f"Source not found: {source_dir}")
            continue
        
        logger.info(f"Processing: {source_dir}")
        
        for output_file in source_dir.glob("*"):
            if not output_file.is_file():
                continue
            
            patient_id = extract_patient_id(output_file.name)
            if not patient_id:
                logger.warning(f"Could not extract patient ID from: {output_file.name}")
                continue
            
            # Create output filename (remove _summary suffix and ensure .txt)
            output_filename = f"{patient_id}.txt"
            output_path = OUTPUT_TOTAL / output_filename
            
            # Check for duplicates
            if patient_id in patient_files:
                logger.warning(
                    f"Duplicate patient ID {patient_id}: "
                    f"keeping {patient_files[patient_id].name}, "
                    f"skipping {output_file.name}"
                )
                duplicate_count += 1
                continue
            
            # Copy file
            try:
                shutil.copy2(output_file, output_path)
                patient_files[patient_id] = output_path
                file_count += 1
            except Exception as e:
                logger.error(f"Error copying {output_file}: {e}")
    
    logger.info(f"✓ Processed {file_count} output files")
    logger.info(f"  Duplicates skipped: {duplicate_count}")
    return patient_files


def validate_matching(input_patients: Dict[str, Path], 
                     output_patients: Dict[str, Path]) -> Tuple[int, int, int]:
    """
    Validate that each input has a matching output and vice versa
    Returns: (total_matched, missing_outputs, missing_inputs)
    """
    logger.info("\nValidating input/output matching...")
    
    input_ids = set(input_patients.keys())
    output_ids = set(output_patients.keys())
    
    matched = input_ids & output_ids
    missing_outputs = input_ids - output_ids
    missing_inputs = output_ids - input_ids
    
    # Report results
    logger.info(f"\n{'='*60}")
    logger.info(f"VALIDATION RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"Total input files:    {len(input_ids)}")
    logger.info(f"Total output files:   {len(output_ids)}")
    logger.info(f"✓ Matched pairs:      {len(matched)}")
    logger.info(f"✗ Missing outputs:    {len(missing_outputs)}")
    logger.info(f"✗ Missing inputs:     {len(missing_inputs)}")
    logger.info(f"{'='*60}\n")
    
    if missing_outputs:
        logger.warning("Inputs without matching outputs:")
        for patient_id in sorted(missing_outputs):
            logger.warning(f"  - {patient_id}")
    
    if missing_inputs:
        logger.warning("Outputs without matching inputs:")
        for patient_id in sorted(missing_inputs):
            logger.warning(f"  - {patient_id}")
    
    # Save detailed report
    report_path = FINAL_DIR / "merge_validation_report.txt"
    with open(report_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("DATA MERGE & VALIDATION REPORT\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Total input files:    {len(input_ids)}\n")
        f.write(f"Total output files:   {len(output_ids)}\n")
        f.write(f"Matched pairs:        {len(matched)}\n")
        f.write(f"Missing outputs:      {len(missing_outputs)}\n")
        f.write(f"Missing inputs:       {len(missing_inputs)}\n\n")
        
        if missing_outputs:
            f.write("INPUTS WITHOUT MATCHING OUTPUTS:\n")
            f.write("-" * 60 + "\n")
            for patient_id in sorted(missing_outputs):
                f.write(f"{patient_id}\n")
            f.write("\n")
        
        if missing_inputs:
            f.write("OUTPUTS WITHOUT MATCHING INPUTS:\n")
            f.write("-" * 60 + "\n")
            for patient_id in sorted(missing_inputs):
                f.write(f"{patient_id}\n")
            f.write("\n")
        
        f.write("MATCHED PAIRS:\n")
        f.write("-" * 60 + "\n")
        for patient_id in sorted(matched):
            f.write(f"{patient_id}\n")
    
    logger.info(f"Detailed report saved to: {report_path}")
    
    return len(matched), len(missing_outputs), len(missing_inputs)


def main():
    """Main execution"""
    logger.info("Starting data merge and validation process...\n")
    
    try:
        # Step 1: Create folder structure
        setup_folders()
        logger.info("")
        
        # Step 2: Process input files
        input_patients = process_input_files()
        logger.info("")
        
        # Step 3: Process output files
        output_patients = process_output_files()
        logger.info("")
        
        # Step 4: Validate matching
        matched, missing_out, missing_in = validate_matching(input_patients, output_patients)
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("MERGE COMPLETE")
        logger.info("="*60)
        logger.info(f"Input files saved to:  {INPUT_TOTAL}")
        logger.info(f"Output files saved to: {OUTPUT_TOTAL}")
        logger.info(f"Report saved to:       {FINAL_DIR / 'merge_validation_report.txt'}")
        logger.info("="*60)
        
        return 0
    
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
