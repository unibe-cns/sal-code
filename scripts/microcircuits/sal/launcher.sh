#!/usr/bin/env sh

# fname stem of the param file:
FNAME="sal"

# Read the content of num_sims.txt to determine the number of simulations
NUM_SIMS=$(cat num_sims.txt)

# Check if the file was read successfully
if [ $? -eq 0 ]; then
  # Check if NUM_SIMS is a valid integer
  if [[ $NUM_SIMS =~ ^[0-9]+$ ]]; then
    echo "Number of sims: $NUM_SIMS"
  else
    echo "Error: The content of num_sims.txt is not a valid integer."
    exit 1
  fi
else
  echo "Error: Unable to read num_sims.txt"
  exit 1
fi

# read the content of res_path.txt to determine where to store the results.
RES_PATH=$(cat res_path.txt)
echo "$RES_PATH"
if [ -d "$RES_PATH" ]; then
    echo "Store the results in $RES_PATH."
else
    echo "Error: The results directory $RES_PATH does not exist."
    exit 1
fi

# run the parameter sweep in parallel
for i in $(seq 0 $NUM_SIMS); do
    python ../run.py $(printf "$RES_PATH/$FNAME.%04i.yaml " $i) $i &
done

wait
