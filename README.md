# SONIA - sensuous original network invents audio

Here you can find implemented RNN that composes music.


For training this RNN I used samples in boogie-woogie style. You can find it in /Samples directory. 


You can check generated samples here: https://drive.google.com/open?id=1Ny8rjikoj6Xp7_9eKGzicqCz4fMV-9Bq


Or try to create music by yourself!


For using it the following libraries need to be installed:

 `numpy`, `midi2audio`, `mido`, `torch`, `os`
 
 (You can download them using `pip3 install name_of_library`).

After all libraries installed:

1) clone repository

2) run `python3 run.py` from Sonia directory

3) enter any integer number

4) now you can find 2 files in .midi and .wav format at current directory


This project was implemented basing on "This Time with Feeling: Learning Expressive Musical Performance" article (https://arxiv.org/abs/1808.03715).
