# script to execute the parallel ssh command in the stream_video bash script using the hosts in the pssh_hosts file

parallel-ssh -i -t 0 -h pssh_hosts ./stream_video.sh
