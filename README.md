# General-synthesis-with-a-WaveNet-and-MFCCs
A TensorFlow implementation of DeepMind's WaveNet with Mel Frequency Cepstrum Coefficients as auxiliary input

Instructions to run:

•	Make sure software libraries installed<br/>
◦	Librosa, Numpy, TensorFlow GPU, SciPy, Matplotlib<br/>

•	Create audio data<br/>
•	Recommend one hour in length in multiple files of ~5 mins each.<br/>
•	Files should be mono, but if multi-channel then the program will automatically take the first channel.<br/>
•	Audio files should be in a seperate folder containing only the audio files that you want to train on.<br/>

•	Generate norms<br/>
◦	Numpy arrays in .np files will be created that will be used to mean subtract and normalised to unit variance the MFCCs.<br/>
◦	Run “python calculate_norm.py --data_dir=audiofolder --output_dir=outfolder”<br/>
◦	The other arguments, sample_rate, hop_length, n_fft, n_mfcc, can be left as default but if you wish to change these you must use the same setting for training and generation.<br/>

•	Train wavenet:<br/>
◦	Run “python train.py --use_aux_features=True --mean_sub=True --normalise=True --mean_file=./meanFile.npy --std_file=./stdFile.npy --data_dir=./directoryOFAudioFiles --save_to=./checkpointDir/ --num_steps=trainingSteps”<br/>
◦	If you wish pause training, enter control-c, and if you wish to continue training from a stored checkpoint remove the argument “--save_to” and add “--restore_from=directoryWithCheckpoints”<br/>

•	Generate audio:<br/>
◦	Run “python generate.py --use_aux_features=True --mean_sub=True --normalise=True --mean_file=./meanfile.npy --std_file=./stdfile.npy --checkpoint=./checkpointFile –samples=numSamplesToGenerate --aux_source=./AudioFileToUseAsAuxSource –wav_out_path=nameOfAudioFile.wav”<br/>
◦	Note: for the checkpoint file, provide only the prefix of the checkpoint and not any of the extensions, i.e. model.ckpt-9999, not model.ckpt-9999.index<br/>

Credit this project (https://github.com/ibab/tensorflow-wavenet) provided a helpful reference.
