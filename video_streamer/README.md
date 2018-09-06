How to use this video streamer:

Everything you need is packaged into the jar
To use stream.sh, you can use
  chmod u+x stream.sh
to make the file excutable and then run it using:
  ./stream.sh

If you're not me, you will have to change the command in the file.

The complete command is this:

java -jar AUV-Streamer.jar nvidia 10.0.0.50 22 nvidia /home/nvidia/usc-computer-vision-2018/output.avi /home/pedro/Videos/camera_test/output.avi /usr/bin/vlc

Note that there should be two changes to the command:
  - the directory to home/pedro is where you want the video to be stored this will vary on your system
  - /usr/bin/vlc points to the vlc installation, this should work for ubuntu systems, otherwise, change this to match your system

The command to run the jar, assuming you have java is:

  java -jar AUV-Streamer.jar [arguments]

There are a few arguments that need to be passed in to connect to the
jetson
These are, in order:
username
host
port
password
path to video on the jetson
path to where you want to put the video on your machine
path to vlc installation on your machine

For use with the jetson, the first five are:
nvidia
10.0.0.50
22
nvidia
/home/nvidia/usc-computer-vision-2018/output.avi

The last two, as explained above depend on your machine.
