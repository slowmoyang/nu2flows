
# check if PROJECT_ROOT is set
if not set -q PROJECT_ROOT
    echo "Error: PROJECT_ROOT environment variable is not set." >&2
    exit 1
end

set project nu2flows-reproduce

for run in (seq 0 2)
    # random number
    set seed (python -c "import random; print(random.randint(1, 1000000))")

    # zero pad the run number
    set run run-$(printf "%02d" $run)

    condor_submit ./submit.condor project=$project run=$run seed=$seed
end


