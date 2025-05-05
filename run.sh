#!/bin/bash

# Create the output_logs directory if it doesn't exist
mkdir -p output_logs

# Loop through the parameters
for iet in {0..4}; do
  for ieta in {0..4}; do
    echo "▶️ Running: iet=$iet, ieta=$ieta"

    # Define a filename for each run's log based on the iet and ieta values
    logfile="output_logs/output_iet${iet}_ieta${ieta}.log"

    # Run the script, saving the output and error to the individual log file and the big log
    if ! python train.py --iet "$iet" --ieta "$ieta" >> "$logfile" 2>> "$logfile"; then
      echo "❌ Failed: iet=$iet, ieta=$ieta" >> errors.log
    fi

    # Append the output and error of the current run to the big log
    cat "$logfile" >> output_logs/big_output.log
  done
done
