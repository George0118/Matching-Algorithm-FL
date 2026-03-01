@echo off

rem Define the program executable path
set program=python

rem Loop through each triple of arguments
for %%a in (12 15 18 21 24 27 30) do (
    for /l %%i in (1,1,10) do (
        rem Run the program with the current set of arguments
        echo Running %program% ./regret_matching-fl-areas.py with arguments: --a 0 --n %%a --k 3 Run %%i
        %program% ./regret_matching-fl-areas.py --a 0 --n %%a --k 3 --tc 8
    )
)

echo All runs completed.