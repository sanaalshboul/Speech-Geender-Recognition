# Speech-Geender-Recognition

Gender recognition using voice is trained on Speech Accent Archive dataset [(download)](https://www.kaggle.com/rtatman/speech-accent-archive?select=speakers_all.csv "download"). The dataset consists of 2135 audio files; 1101 of them are males and 1034 are females. The participants’ ages range between 6-97 years and the following figure shows the distribution of the number of participants among ages. The number of languages that was used for collecting the dataset is 201 languages. One utterance [(Utterance)](https://github.com/sanaalshboul/Speech-Geender-Recognition/blob/master/reading-passage.txt "download") was used for recording the voice with utterance length of 73 words. 

  Info        |     Number    |
------------- | ------------- | 
 Females      |       1034    |     
 Males        |       1101    |    
 languages    |       201     | 
 Utterances   |       1       | 
 
 ![picture alt](https://github.com/sanaalshboul/Speech-Geender-Recognition/blob/master/images/age.png "Title is optional")
 
 ## Requirments
* Pandas
* Numpy
* Matplotlib
* Sklearn
* Scipy
* Librosa
* Keras
* Tensorflow
* Xgboost

## Data cleaning
[(audio dataset)](https://github.com/sanaalshboul/Speech-Geender-Recognition/tree/master/recordings/recordings "download") contains 1000 audios of the dataset, because Github allows 1000 files to be submitted in one folder. You can download the dataset from this link [(download)](https://www.kaggle.com/rtatman/speech-accent-archive?select=speakers_all.csv "download").

Run [(Train_Test.ipynb)](https://colab.research.google.com/github/sanaalshboul/Speech-Geender-Recognition/blob/master/Train_Test.ipynb "download"), because it contains all steps from cleaning data to training.

[(speakers_all.csv)](https://github.com/sanaalshboul/Speech-Geender-Recognition/blob/master/speakers_all.csv "download") contains all information about the recorded audios. So, before training the dataset it should be cleaned by removing the rows where:
* The audio file is removed from the dataset ('file_missing' =True). 
* The age equals zero ('age'=0) because there is no person has age zero.
* The gender is not male or female.

## Explanatory voice analysis
The left figures in the following figures show male and female audio ploting. While the middle figures present the difference between male and female spectrogram, which produces the intensities of accuring freuencies over the time in time-frequency domain. The lighter region in the spectogram plot indicates higher frequencies. In female audio the frequency is higher than in male audio, which is centered in the frequecny range 0-8000. The right figures produce the magnitude spectrum for male and female audios, which produces the audio energy. The magnitude spectrum or the energy of the male audio is centered in higher energy than in female audio.
 ![picture alt](https://github.com/sanaalshboul/Speech-Geender-Recognition/blob/master/images/spectrum.png "Title is optional")

## Feature Extraction 
`feature_extraction(file_name, featuer_name)` method is used to extract the voice features for every audio file. In this work 'mel' feature is extracted for the all audios and saved in [(all_audio_features.csv)](https://github.com/sanaalshboul/Speech-Geender-Recognition/blob/master/data/all_audio_features.npy "download") numpy array file, while the gender for all audios are saved in [(all_labels.csv)](https://github.com/sanaalshboul/Speech-Geender-Recognition/blob/master/data/all_labels.npy "download") file, so you don't need to extract 'mel' feature another time. Another features (such as 'chroma', 'contrast', 'tonnetz') can be extracted and saved by calling `feature_extraction(file_name, featuer_name)` for every audio file. Ex: `feature_extraction(file_name, 'chroma')`.

## Data splitting, training, and testing
The dataset is divided to 70% train, 15% validation, and 15% testing sets.

Two models are used to train the dataset; MLPClassifier and Neural Network (NN) model. The hyperparameters of each model are choosen based on hyperparameters tuning using GridSearchCV algorithm. The test result of both models are shown in this table. NN model achieved higher accuracy (95%), which mean that 95% of the time the model is able to detect if the speaker is male or female. On the other hand, MLPClassifier acheived 93% accuracy score, 93 precision, and 93 recall. Precision is a measurment for the accurate positive predictions to all positive prediction, so higher precision is better. Recall measures the propotion of accurate positive predictions out of all actual positive observations.

Algorithm        | Accuracy score  |  Precision  | Recall  |
---------------- | --------------- | ----------- | ------- |
MLPClassifier    |       93%       |     93      |   93    |
NN               |       95%       |     --      |   --    |
