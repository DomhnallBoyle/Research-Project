# script to ssh to all the PIs and run a command to shut them down

ssh pi@10.42.0.44 'sudo shutdown -h now'
echo 10.42.0.44 - Left shutdown successfully

ssh pi@10.42.0.15 'sudo shutdown -h now'
echo 10.42.0.15 - Center shutdown successfully

ssh pi@10.42.0.11 'sudo shutdown -h now'
echo 10.42.0.11 - Right shutdown successfully
