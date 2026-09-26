@echo off
cd /d "%~dp0"

echo ============================================
echo A1: TFFM branch-only ablation (structural losses = 0)
echo ============================================
python run_ablation.py config_topology_ablation_branch_only.yaml
if errorlevel 1 goto :error

python -m src.evaluate_topology_v2 --config config_topology_ablation_branch_only.yaml --checkpoint checkpoints\topology_branch_only_seed42\best_model.pth --datasets DRIVE_test STARE CHASE_DB1 HRF --output-dir evaluation_outputs\ablation_A1_branch_only_no_tta
if errorlevel 1 goto :error

python -m src.evaluate_topology_v2 --config config_topology_ablation_branch_only.yaml --checkpoint checkpoints\topology_branch_only_seed42\best_model.pth --datasets DRIVE_test STARE CHASE_DB1 HRF --tta --output-dir evaluation_outputs\ablation_A1_branch_only_tta
if errorlevel 1 goto :error

echo ============================================
echo A2: TFFM + clDice-only ablation
echo ============================================
python run_ablation.py config_topology_ablation_cldice_only.yaml
if errorlevel 1 goto :error

python -m src.evaluate_topology_v2 --config config_topology_ablation_cldice_only.yaml --checkpoint checkpoints\topology_cldice_only_seed42\best_model.pth --datasets DRIVE_test STARE CHASE_DB1 HRF --output-dir evaluation_outputs\ablation_A2_cldice_only_no_tta
if errorlevel 1 goto :error

python -m src.evaluate_topology_v2 --config config_topology_ablation_cldice_only.yaml --checkpoint checkpoints\topology_cldice_only_seed42\best_model.pth --datasets DRIVE_test STARE CHASE_DB1 HRF --tta --output-dir evaluation_outputs\ablation_A2_cldice_only_tta
if errorlevel 1 goto :error

echo ============================================
echo ALL ABLATION RUNS COMPLETE
echo ============================================
goto :eof

:error
echo.
echo *** A STEP FAILED - check the output above, do not proceed ***
exit /b 1
