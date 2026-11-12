@echo off
rem The second, full benchmark run with the final code (12 November 2026), in its own "v2" folders:
rem   1. rebuild the frozen Experiment_Memory (30 Sessions on the development datasets, about 1.75 h)
rem   2. freeze it to results\benchmark-v2\frozen_memory.jsonl (skipped if that file exists: never rewritten)
rem   3. the smoke setting (every system once on titanic, seed 1, 120 s)
rem   4. the whole grid (390 cells on the evaluation datasets, about a day)
rem Every step skips the work already done, so running this file again after an interruption continues
rem where it stopped. Logs: experiments\benchmark-v2-logs\. Keep the laptop on mains power.
setlocal
cd /d "%~dp0.."
set MLFLOW_DISABLE_AGENT_HINT=1
set PY=C:\Users\priya\miniconda3\envs\automl\python.exe
set LOG=experiments\benchmark-v2-logs
set MEMORY=results\benchmark-v2\frozen_memory.jsonl
if not exist "%LOG%" mkdir "%LOG%"

echo [%date% %time%] started >> "%LOG%\chain.log"
echo [%date% %time%] memory build >> "%LOG%\chain.log"
"%PY%" -u scripts\build_memory.py run --label v2 >> "%LOG%\memory_build.log" 2>&1
if errorlevel 1 goto failed
if exist "%MEMORY%" goto smoke
echo [%date% %time%] freeze >> "%LOG%\chain.log"
"%PY%" -u scripts\build_memory.py freeze --label v2 >> "%LOG%\memory_build.log" 2>&1
if errorlevel 1 goto failed

:smoke
echo [%date% %time%] smoke >> "%LOG%\chain.log"
"%PY%" -u scripts\run_benchmark.py run --smoke --label v2 --memory "%MEMORY%" >> "%LOG%\smoke.log" 2>&1
if errorlevel 1 goto failed
echo [%date% %time%] grid >> "%LOG%\chain.log"
"%PY%" -u scripts\run_benchmark.py run --label v2 --memory "%MEMORY%" >> "%LOG%\grid.log" 2>&1
if errorlevel 1 goto failed
echo [%date% %time%] DONE >> "%LOG%\chain.log"
goto end

:failed
echo [%date% %time%] FAILED with exit code %errorlevel% (see the step's log) >> "%LOG%\chain.log"
:end
endlocal
