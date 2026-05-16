#!/usr/bin/env python3
"""
GeoClimate-FusionLLM: Master Pipeline
Run all steps sequentially to reproduce paper results.

Usage: python scripts/run_all.py
"""
import subprocess, sys, time, os

SCRIPTS = [
    ("01_feature_engineering.py", "Feature Engineering (raw → 159 features)"),
    ("02_train_baselines.py",     "Training Classical + Neural Baselines"),
    ("03_train_mmwstm_adran.py",  "Training MMWSTM-ADRAN+ Ensemble"),
    ("04_evaluate.py",            "Evaluation, DM Tests, Ablation, Robustness"),
    ("05_generate_figures.py",    "Generating Publication Figures"),
]

def main():
    print("=" * 70)
    print("GeoClimate-FusionLLM: Full Reproducibility Pipeline")
    print("=" * 70)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    total_start = time.time()
    
    for i, (script, desc) in enumerate(SCRIPTS, 1):
        print(f"\n{'='*70}")
        print(f"STEP {i}/{len(SCRIPTS)}: {desc}")
        print(f"{'='*70}")
        
        path = os.path.join(script_dir, script)
        start = time.time()
        result = subprocess.run([sys.executable, path], capture_output=False)
        elapsed = time.time() - start
        
        if result.returncode != 0:
            print(f"\n*** ERROR in {script} (exit code {result.returncode}) ***")
            sys.exit(1)
        print(f"\n  Completed in {elapsed:.1f}s")
    
    total = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"ALL STEPS COMPLETED in {total:.1f}s ({total/60:.1f} min)")
    print(f"{'='*70}")
    print(f"\nOutputs:")
    print(f"  Results: results/paper_results.json")
    print(f"  Figures: figures/*.png")
    print(f"  Paper:   paper/GeoClimate_FusionLLM_R2.docx")

if __name__ == "__main__":
    main()
