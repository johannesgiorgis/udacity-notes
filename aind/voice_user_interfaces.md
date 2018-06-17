# AIND - Voice User Interfaces Notes
_This contains notes and references from Udacity AIND Term 2's Voice User Interfaces section._


[Lesson Plan](https://sites.google.com/knowlabs.com/aind-student-resources/lesson-plans/term-2-voice-user-interfaces)

## Lesson 22: Intro to Voice User Interfaces

#### Welcome to Voice User Interfaces!
- VUI is a speech platform that enables humans to communicate with machines by voice
- Under the hood of a VUI platform, there is a lot going on:
	- Audio sound waves from voices must be converted into text using machine learning algorithms and probabilistic models
	- The resulting text must be reasoned over using AI logic to determine the meaning and formulate a response
	- Finally the response text must be converted back into understandable speech again with machine learning tools
	- These 3 parts constitute a general pipeline for building an end to end voice enabled application
- Go through a VUI's system overview and some current VUI applications, focusing on conversational AI applications

#### VUI Overview
- 3 general pieces of the VUI pipeline:
	- Voice to Text
	- Text Input reasoned to Text Output
	- Text to Speech
- Voice to Text is Speech Recognition
	- Speech Recognition easy for people but difficult for machines
	- As a person speaks into a microphone, sound vibrations are converted into an audio signal
	- This signal is sampled at some rate and those samples are converted into vectors of component frequencies
	- These vectors represent features of sound in a data set
	- This step can be considered a feature extraction step
	- The next step is to recognize the series of vectors as a word or sentence
	- To do this, we need probabilistic models that work well with time series data for the sound patterns
	- This is the Acoustic model
	- Decoding the vectors with an acoustic model will give us a best guess as to what the words are 
	- Need to add a Language Model and maybe an Accent Model to our decoder
	- Acoustic, Language and Accent models are all needed for a robust system
	- Built Hidden Markov Models (HMMs) to decode a series of gestures
	- Used Recurrent Neural Networks (RNNs) to train time series data
	- Both these models have been used successfully in Speech Recognition
- Once our speech is in text form, we need to make sense of it - thinking part, the reasoning logic
	- In order to come up with a response, you need to understand what the request was, then process the request and formulate a response
	- This is the field of Natural Language Processing
	- To fully implement NLP, large datasets of language must be processed along with a great deal of challenges to overcome
	- We can focus on a smaller problem: getting a weather report from a VUI device
	- Rather than parsing all the words, we take a shortcut and map the most probable request phrases for the weather to get weather process
	- In this case, the application would understand requests most of the time
	- This won't work if a request hasn't been premapped as a possible choice, but is quite effective for limited applications and can be improved over time
- Once we have a text response, the remaining task is to convert that text to speech
	- Examples of how words are spoken can be used to train a model, to provide the most probable pronunciation components of spoken words
	- Complexity of the task varies greatly when we move from a monotonic robotic voice to a rich human sounding voice that includes inflection and warmth
	- Some of the most realistic sounding machine voices to date have been produced using deep learning techniques


#### VUI Applications
- VUI Applications are becoming more and more common place
- Few reasons driving this:
	- Voice is natural for humans
	- Voice is fast - speaking into a text transcriber is three times faster than typing
	- There are times when it is too distracting to look at a visual interface like when you're walking or driving
- Better and accessible speech recognition and speech synthesis technologies, a number of applications have flourished:
	- Voice interfaces in cars helping the driver get more done while being focused on the road
	- Other web and mobile applications are getting better
	- Dictation applications leverage speech recognition technologies to make putting thoughts into words a snap
	- Translation applications leverage speech recognition and speech synthesis as well as some reasonable logic in between to convert speech from one language to another
	- An exciting innovation is conversational AI technology

#### What is an Alexa Skill?
- Alexa is Amazon’s voice service and the brain behind millions of devices including the Amazon Echo. Alexa provides capabilities, or skills, that enable customers to create a more personalized experience. There are now more than 12,000 skills from companies like Starbucks, Uber, and Capital One as well as innovative designers and developers.

#### Conversational AI with Alexa
- One of the most popular application areas for voice systems
- Originally inspired by the Star Trek computer
- In the past, voice systems on phones step you through the menu but under some circumstances that can be frustrating
- This is a graph based interaction:
	- Focuses on asking pointed questions in a prescribed order and only accepting specific terms as responses
	- e.g. you can't move forward in the system until you provide the your user ID or you can't specify your destination until you provide your starting location
- Alternative frame based interaction allows the user to drive the interaction:
	- You can say the words that make sense to you
	- You can make requests in the order you prefer when it makes sense
	- You can jump to the part of the menu you want to be in without having to memorize a list of options
	- e.g. 'I want to fly from Columbus to San Francisco on October 9th'
	- Alexa will understand this full request and confirm with you
- This is exciting as it allows for flexibility in what we can build, when we design our own Alexa skills
- How do we know what we want to have Alexa listen for? And how does that translate to what we want in response?
	- You defined a series of actions the user can perform which are called Intents
	- e.g. DVD Player Voice Skill: intents might be play, pause, stop, eject
	- DVD Player doesn't have a pizza button because DVD players don't make pizzas
	- Create set of sample statements called utterances that help Alexa understand which intent to use when a user says something
	- For DVD Player, someone might say "start the movie" and expect the play intent to happen
	- They might also say "play", "go", "begin" or "it's showtime!" and expect the same reaction: the intention is to play the movie
- More user testing could help us expand our list of better answers as well
- How do we model what the user will say?
	- Grab colleagues, friends and complete strangers and ask them to naturally request what outcome they need
	- Discover that each person approaches it differently and your skills need to accommodate all of them
	- e.g. to ask Alexa the weather: "What is the temperature outside?", "How hot is it?", "What's it like?"
	- There are lots of ways to indicate that you want to know the weather without saying it
- Are there some other guidelines we need to keep in mind for a good user experience?
	- Number of ideas to consider when designing a voice only interface
	- They are different from the design of text or graphical interfaces


#### VUI Best Practices
- What best practices should we think about when building our Alexa skill?
	- A voice only interface puts a cognitive load on the user
	- We can't expect the user to remember a long list of prompts from Alexa for more information
	- Requests for information from users needs to be clear and brief without being cryptic
	- Users want to know they have been understood without each word being parroted back to them
	- Need to observe a careful balance
- It's important for us as developers to think of ways to subtly provide this verification
	- e.g. ask Alexa to start timer
	- "Alexa, start Timer"
	- Alexa: "Timer for how long?"
	- "One hour"
	- Alexa: "One hour, starting now"
	- You know you were correctly heard based on Alexa's responses, yet the word timer was never included in Alexa's responses
	- You were asked only for additional information that was truly needed to start the timer
- There may be applications where you need very clear verification before action: e.g. transferring money from one account to another
	- You need to do what makes sense for your users and your skill
	- When dealing with finances, it is beneficial to be more predictable with your interactions
- What else can we do to make our Alexa skills easy to interact with?
	- Make your skill conversational
	- Randomize how and when Alexa asks for information
	- Include five to seven statements that have the same meaning
- Randomization is a key to making conversations appear more natural
- Summary
	- Provide a large number of utterances to account for randomized requests from humans
	- Randomizing the order of requests for information by Alexa
	- Randomizing equivalent responses from Alexa
- Different way of thinking about the user interface compared to non-voice applications
- What about error handling? If I need help? If I want Alexa to stop?
	- Skills should include some typical Built-in Intents such as help, stop, and cancel
	- Other Built-in Intents: Yes, No, Start Over
	- Each one is a pre-trained intent that users will expect to work a specific way in your skill in the same way that apps on your phone have some consistent behaviors


#### Lab: Space Geek
- Use an existing Alexa template for a fact skill named "space geek"

#### Alexa Skills - Beyond Space Geek
- In a fact skill, user can only ask for and receive one fact
- For more advanced skills, you sometimes want to know something about what the user said like a specific address or a person's name
- Slots provide a way for your to create utterances that have specific holes in them for those values to travel to your code
- Create your own list of slot values or use some of the built-in slot values that Amazon provides
- How can we take a basic one-liner type skill and make it more conversational?
	- Fact skill has a tell statement to tell the user something and end the conversation
	- Replace the tell statement with an ask statement and Alexa will wait for another response from the user
	- Make sure to ask the user a question so they know they have an opportunity to continue the conversation
	- After the question, Alexa waits for 8 seconds for user to respond
	- If she doesn't hear anything, you can provide a re-prompt statement to remind the user that Alexa is waiting for a response
	- There are tons of other more advanced things that you can do with your code:
		- persisting to databases
		- making HTTPS request across the web
		- managing the state of your user so that you have context when they respond to your skill
- What else can we get Alexa to do?
	- Support modern programming languages
	- Ability to host your code in the environment of your choice
	- Link to existing user accounts with OAuth
	- control IoT devices with Amazon Smart Home API
	- integrate voice into your existing software and hardware projects


## Lesson 23: Alexa History Skill
_Create your own Alexa History Skill_
- Build a fully functional skill for Amazon's Alexa that provides year-dated facts from AI History.

## Lesson 24: Introduction to Speech Recognition
_Dive deeper into the exciting field of Speech Recognition, including cutting edge deep learning technologies for Automatic Speech Recognition_

#### Intro
- Speech Recognition = ASR, Automatic Speech Recognition:
	- Goal is to input any continuous audio speech and output the text equivalent
	- Want it to be speaker independent and have high accuracy
	- A core goal of AI for a long time
	- In 1980's and 1990's, advances in probabilistic models began to make ASR a reality
- What makes speech recognition hard?
	- Like other AI problems, ASR can be implemented by gathering lots of labeled data, training a model in that data and then deploying the trained model to accurately label new data
	- Twist: speech is structured in time and has a lot of variability
	- Challenges involved with decoding spoken words and sentences into text
- Models in speech recognition can conceptually be divided into an acoustic model and a language model:
	- Acoustic model solves the problem of turning sound signals into some kind of phonetic representation
	- Language model houses the domain knowledge of words, grammar and sentence structure for the language
	- Implemented with probabilistic models using machine learning algorithms
	- Hidden Markov Models (HMMs) have been refined with advances for ASR over a few decades and considered the traditional ASR solution
	- Cutting edge ASR involves end to end Deep Neural Network models


#### Challenges in ASR
- Continuous speech recognition has a rocky history
	- Started in 1970's with US funding ASR research with a DARPA Challenge
	- CMU Harpy 1000 words
	- led to first big AI Winter
	- Performance improved in 1980's and 1990's with better probabilistic models
	- Recently computer power has made larger dimensions in neural network modeling a reality
- Why is Speech Recognition so hard?
	- First, the audio signal itself contains a lot of noise (background noise, cars, clocks, background conversations...etc)
	- ASR has to know which part of the audio signal to focus on and which to ignore
	- Variability of pitch
	- Variability of volume
	- Same words sound different when said by different people
	- Variability of word speed: words spoken at different speeds need to be aligned and matched. e.g. speech and speeeeeeech are the same word.
	- Word boundaries: Words run from one to the next without pause when we speak as we naturally do not separate them
	- Language knowledge: we have domain knowledge of our language, allowing us to sort out ambiguities as we hear them. e.g. "Recognize speech" vs. "Wreck a nice beach"
	- Spoken vs Written Language: there are hesitations, repetitions, fragments of sentences, slips of the tongue that we can filter out
- Summary:
	- Variability in:
		- Pitch
		- Volume
		- Speech
	- Ambiguity:
		- Word Boundaries
		- Spelling
		- Context


#### Signal Analysis
- When we speak, we create sinusoidal vibrations in the air
- Higher pitches vibrate faster with a higher frequency than lower pitches
- These vibrations can be detected by a microphone and transduced from acoustical energy carried in the sound wave to electrical energy where it is recorded as an audio signal
- Need to get a handle on the features that make up our input
- What's going on in a 'Hello World' signal?
	- Two blobs corresponding to the two words - hello, world
	- Some vibrations are taller than others, or have a higher amplitude
	- Amplitude in the audio signal tells us how much acoustical energy is in the sound, how loud it is
	- Looking at a time slice of the signal:
		- It has an irregular wiggle shape to it
		- Our speech is made up of many frequencies at the same time
		- The signal we see is the sum of all these frequencies stuck together
		- Use the component frequencies as features to properly analyze the signal
		- Use a Fourier Transform to break the signal into these components (Fast Fourier Transform algorithm)
		- Use this splitting technique to convert the sound into a Spectrogram
- Spectrogram:
	- Plot Frequency (vertical) against time (horizontal)
	- Intensity of shading indicates the amplitude of the signal
	- To create Spectrogram:
		- Divide signal into time frames
		- Split each frame signal into frequency components with an FFT
		- Each time frame is now represented with a vector of amplitudes at each frequency
		- Lining up the vectors in their time series order results in a visual picture of the sound components, the Spectrogram
	- The Spectrogram can be lined up with the original audio signal in time
	- It provides us with a complete representation of our sound data
	- But data still has noise and variability
	- Data may contain more information than we really need
	- Use feature extraction to reduce the noise and dimensionality of data


#### References: Signal Analysis
- [Sound Wikipedia Page](https://en.wikipedia.org/wiki/Sound)
	- Excellent explanations and definitions for vibration, frequency, sound waves...etc
- [Signal Analysis - Speech Recognition](http://web.science.mq.edu.au/~cassidy/comp449/html/ch03.html)
- [Fourier Transforms](https://ibmathsresources.com/2014/08/14/fourier-transforms-the-most-important-tool-in-mathematics/)
	- Fourier Analysis is the study decomposing mathematical functions into sums of simpler trigonometric functions. Since sound is comprised of oscillating vibrations, we can use Fourier analysis, and Fourier transforms to decompose an audio signal into component sinusoidal functions at varying frequencies.
- [Spectrogram](http://www.seas.upenn.edu/~cis391/Lectures/speech-rec.pdf)
	- A spectrogram is the frequency domain representation of the audio signal through time. It's created by splitting the audio signal into component frequencies and plotting them with respect to time. The intensity of color in the spectrogram at any given point indicates the amplitude of the signal. 

#### Quiz: Fast Fourier Transforms
- [Fast Fourier Transform](https://en.wikipedia.org/wiki/Fast_Fourier_transform)
	- Efficient implementation of a Discrete Fourier Transform
	- The algorithm transforms a sum of sinusoidal signals into into its pure frequency components


#### Feature Extraction with MFCC
- What part of the audio signal is really important for recognizing speech?
	- Our speech is constrained by our voice making mechanisms and what we can perceive with our ears
	- If we can't hear a pitch, no need to include it in our data
	- If we can't distinguish between two different frequencies, consider them as the same frequency for our purposes
	- For feature extraction purposes, put the frequencies of the spectrogram into bins that are relevant to our own ears and filter out sound we can't hear. This reduces the number of frequencies
	- Separate the elements of sound that are speaker independent by focusing on the voice-making mechanism we use to create speech
	- Think of a human voice production model as a combination of source and filter, where source is unique to an individual and filter is the articulation of words that we all use when speaking
	- Cepstral Analysis relies on this model for separating the two
	- The cepstrum can be extracted from the signal via an algorithm [???]
	- The main thing to remember is that we're dropping the component of speech unique to individual vocal chords and preserving the shape of the sound made by the vocal tract
	- Cepstral Analysis combined with MEL Frequency Analysis provides 12 - 13 MFCC features related to speech
	- Delta and Delta-Delta MFCC features can optionally be appended to the feature set, doubling or tripling the number of features (up to 39), which has been shown to give better results in ASR
	- The takeaway from using MFCC feature extraction is that:
		- Reduced dimensionality
		- Reduced noise

#### References: Feature Extraction
- [Summary of Methods Used](http://www.ijcsmc.com/docs/papers/March2015/V4I3201545.pdf)
- [MEL Wikipedia Page](https://en.wikipedia.org/wiki/Mel_scale)
	- The Mel Scale was developed in 1937 and is based on human studies of pitch perception. At lower pitches (frequencies), humans can distinguish pitches better.
- [Source/Filter Model](http://web.science.mq.edu.au/~cassidy/comp449/html/ch07.html#d0e1094)
	- The source/filter model holds that the "source" of voices speech is dependent upon the vibrations initiated in the vocal box, and is unique to the speaker, while the "filter" is the articulation of the words in the forward part of the voice tract. The two can be separated through Cepstrum Analysis.
- [Cepstral Analysis of Speech (Theory)](http://iitg.vlab.co.in/?sub=59&brch=164&sim=615&cnt=1)
- [MFCC](http://www.speech.cs.cmu.edu/15-492/slides/03_mfcc.pdf)
	- Mel Frequency Cepstrum Coefficient Analysis is the reduction of an audio signal to essential speech component features using both mel frequency analysis and cepstral analysis. The range of frequencies are reduced and binned into groups of frequencies that humans can distinguish. The signal is further separated into source and filter so that variations between speakers unrelated to articulation can be filtered away. 
- [MFCC Deltas and Delta-Deltas](http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/)
	- Intuitively, it makes sense that changes in frequencies, deltas, and changes in changes in frequencies, delta-deltas, might also be meaningful features in speech recognition.


#### Quiz: MFCC
- [Mel Frequency Cepstral Coefficients (MFCC)](http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/) feature extraction is complicated to explain, but easy to implement with available libraries
- [scip.io.wavfile.read](https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.io.wavfile.read.html): to extract a signal
- [python_speech_features.mfcc](http://python-speech-features.readthedocs.io/en/latest/): to convert to MFCC


#### Phonetics
- Study of sound in human speech
- Linguistic analysis of language around the world is used to break down human words into their smallest sound segments
- In any given language, some number of phonemes defines the distinct sounds in that language
- In US English, there are 39 - 44 phonemes to find
- Grapheme: Smallest distinct unit that can be written in a language
- In US English, the smallest grapheme set we can define is a set of the 26 letters in the alphabet plus a space
- Can't map phonemes to grapheme or individual letters because some letters map to multiple phonemes sounds and some phonemes map to more than one letter combination
- Example: in English the _C_ letter sounds different in cat, chat, and circle; the phoneme _E_ sound we hear in receive, beet and beat is represented by different combinations
- Arpabet, a US English phoneme set, was developed in 1971 for speech recognition research; It contains 39 phonemes, 15 vowel sounds, 24 consonants, each represented as one or two letter symbol
- Phonemes are often a useful intermediary between speech and text
- If we can successfully produce an acoustic model that decodes a sound signal into phonemes the remaining task would be to map those phonemes to their matching words. This step is called Lexical Decoding and is based on a lexicon or dictionary of the data set.
- Steps:
	- Speech -> Features -> Acoustic Model -> Phonemes -(Lexical Decoding)> Words -> Text
- Why not use our acoustic model to translate directly into words? Why take the intermediary step?
	- e.g. Speech -> Features -> Acoustic Model -> Words -> Text
	- Good question. There are systems that translate features directly into words
	- This is a design choice and depends on the dimensionality of the problem
	- If we are training a limited vocabulary of words, we might just skip the phonemes, but if we have a large vocabulary, converting to the smaller units first reduces the number of comparisons that need to be made in the system overall


### References: Phonetics
- [Phonetics Wikipedia Page](https://en.wikipedia.org/wiki/Phonetics)
	- A branch of linguistics for the study of sounds of human speech: physical properties, production, acoustics, articulation, etc.
- [Phoneme Wikipedia Page](https://en.wikipedia.org/wiki/Phoneme)
	- In any given language, a _phoneme_ is the smallest sound segment that can be used to distinguish one word from another. For example "bat" and "chat" have only one sound different but this changes the word. The phonemes in question are "B" and "CH". What exactly these are and how many exist varies a bit and may be influenced by accents included. Generally, US English consists of 39 to 44 phonemes. See ARPAbet below for more phoneme examples.
- [Grapheme Wikipedia Page](https://en.wikipedia.org/wiki/Grapheme)
	- The definition of a grapheme is somewhat inconsistent in the literature. In our context, a grapheme is the smallest symbol that distinguishes one written word from another. For example, "bat" and "chat" have a difference of two graphemes, even though "CH" is considered to be a single phoneme. In US English, 26 letters and a space combine for 27 possible graphemes.
- Lexicon
	- A lexicon for speech recognition is a lookup file for converting speech parts to words
	- e.g. [cmudict](http://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict/sphinxdict/cmudict_SPHINX_40): the Carnegie Mellon tool for speech recognition compatible with the open source [Sphinx project](https://cmusphinx.github.io/)
- [ARPAbet Wikipedia page](https://en.wikipedia.org/wiki/Arpabet)
	- A set of phonemes developed by the Advanced Research Projects Agency(ARPA) for the Speech Understanding Project (1970's)
	- [ARPAnet dictionary at CMU](http://www.speech.cs.cmu.edu/cgi-bin/cmudict)


#### Quiz: Phonetics
- [Pangram Wikipedia Page](https://en.wikipedia.org/wiki/Pangram)
	- A sentence using every letter in a given alphabet at least once
	- A _phonetic pangram_ is a sentence that uses every _phoneme_ at least once
	- Phonetic Pangrams:
		- The phoneme pangrams are a bit longer because they must include more elements (the phonemes) than just 26 different letters.
		- The beige hue on the waters of the loch impressed all, including the French queen, before she heard that symphony again, just as young Arthur wanted
	- The phoneme pangrams are a bit longer because they must include more elements (the phonemes) than just 26 different letters.


#### Voice Data Lab Introduction
- 
- TIMIT Acoustic-Phonetic Corpus
	- A popular benchmark data source for AST training testing
	- Data was developed specifically for speech research in 1993 and contains 630 speakers voicing 10 phoneme-rich sentences each
- 2 Popular large vocabulary data sources:
	- LDC Wall Street Journal Corpus (73 hours of newspaper reading)
	- LibriSpeech (1000 hours of readings from public domain books)
- Tools for converting these various audio files into Spectrograms and other feature sets are available in a number of software libraries
- Explore some dataset samples as well as create some audio files and data of your own; Can also take a look at the spectrograms with an open-source visualization tool
- [AIND-VUI-Lab-Voice-Data Repository](https://github.com/udacity/AIND-VUI-Lab-Voice-Data)
- [LibriSpeech Corpus Website](http://www.openslr.org/12/)


#### Acoustic Models and Trouble with Time
- At the conclusion of the Voice Data Lab, we have our data
- With feature extraction, we have addressed noise problems due to environmental factors as well as variability of speakers
- Phonetics gives us a representation for sounds and language that we can map to
- That mapping, from the sound representation to the phonetic representation is the task of our acoustic model
- Still have not solved the problem of matching variable lengths of the same word
- Dynamic Time Wrapping algorithm:
	- Calculates the similarity between two signals even if their times lengths differ
	- Used in speech recognition to align the sequence data of a new word to its most similar counterpart in a dictionary of word examples
- Hidden Markov Models are well-suited for solving this type of time series pattern sequencing within an acoustic model. This characteristic explains their popularity in speech recognition solutions for the past 30 years.
- If we choose to use deep neural networks for our acoustic model, the sequencing problem reappears. We can address the problem with a hybrid HMM/DNN system or we can solve it another way
- Later discuss how we can solve this problem in DNNs with Connectionist Temporal Classification (CTC)


#### HMMs in Speech Recognition
- HMMs are useful in detecting patterns through time, which is what we are trying to do with an acoustic model
- HMMs can solve the challenge of time variability - e.g. "speech" vs. "speeeeeeech"
- Train an HMM with label time series sequences to create individual HMM models for each particular sound unit. Units can be phonemes, syllables, words, or groups of words
- Training HMMs with isolated units is straight forward
- Training HMMs with continuous speech is challenging
- How can the series of phonemes or words be separated by training?
	- Utterances: continuous phrases or sentences
	- Tie HMMs together as pairs
	- Dimensionality increases: We need an HMM for each word and each possible word connection
	- Phonemes require less dimensionality space than words for large vocabularies
	- Example: With 40 phonemes, we need 1600 HMMs to account for the transitions
	- Once trained, the HMM models can be used to score new utterances through chains of probable paths


####  Language Models
- So far, we have tools for addressing noise and speech variability through our feature extraction
- HMM models convert those features into phonemes and address sequencing problems for our full acoustic model
- Yet to solve the problems in language ambiguity
- ASR system can't tell from the acoustic model which combinations of words are most reasonable
- Need to either provide that knowledge to the model or give it a mechanism to learn this contextual information on its own


#### N-Grams
- Language Model's job is to inject language knowledge into the words to text step in speech recognition, providing another layer of processing between words and text to solve ambiguities in spelling and context
	- Speech -> Features -> Acoustic Model -> Phonemes -> Words -(Language Knowledge)> Text
	- Speech -> Features -> Acoustic Model -> Phonemes -> Words -> Language Model -> Text
- Example: since an acoustic model is based on sound, we can't distinguish the correct spelling for words that sound the same - "here" vs "hear"
- Other sequences may not make sense but could be corrected with a little more information
- Words produced by the Acoustic Model are not absolute choices:
	- Thought of as a probability distribution over many different words
	- Each possible sequence can be calculated as the likelihood that the particular word sequence could have been produced by the audio signal
	- A statistical language model provides a probability distribution over sequences of words
	- With the Acoustic and Language models together, the most likely sequence would be a combination over all these possibilities with the greatest likelihood score
	- If all possibilities in both models were scored, this could be a very large dimension of computations
	- Get a good estimate by looking at some limited depth of choices
	- In practice, the words we speak at any time are primarily dependent upon only the previous 3 - 4 words
- N-Grams:
	- Probabilities of single words, ordered pairs...etc
	- Approximate the sequence probability with the chain rule
	- The probability that the first word occurs is multiplied by the probability of the second given the first and so on to get probabilities of a given sequence
	- Score these probabilities along with the probabilities from the Acoustic model to remove language ambiguities from the sequence options and provide a better estimate of the utterance in text


#### Quiz: N-Grams
- N-Gram is an ordered sequence of words:
	- 1-gram "I"
	- 2-gram "I love"
	- 3-gram "I love language"
	- 4-gram "I love language models"
- [Bigrams Wikipedia Page](https://en.wikipedia.org/wiki/Bigram)
- [Markov Assumption Wikipedia Page](https://en.wikipedia.org/wiki/Markov_property)
- [Laplace Smoothing Wikipedia Page](https://en.wikipedia.org/wiki/Additive_smoothing)


#### References: Traditional ASR
- [Computer History Museum Video on Traditional ASR](https://www.youtube.com/watch?v=PJ_KCTsOCrs)
- Acoustic Models with HMMs:
	- HMMs are the primary probabalistic model in traditional ASR systems. The following slide decks from Carnegie Mellon include very helpful and detailed visualizations of HMM's, the Viterbi Trellis, State Tying, and more from the Carnegie Mellon:
	- [slides - HMMs](http://www.cs.cmu.edu/~bhiksha/courses/11-756.asr/spring2014/lectures/class7-8.hmm.pdf)
	- [slides - Continuous Speech](http://www.cs.cmu.edu/~bhiksha/courses/11-756.asr/spring2014/lectures/class9.continuousspeech.pdf)
	- [slides - HMM tying](http://asr.cs.cmu.edu/spring2011/class21.6apr/class21.subwordunits.pdf)
- N-Grams:
	- N-Grams provide a way to constrain a series of words by chaining the probabilities of the words that came before.
	- [Speech and language processing](https://lagunita.stanford.edu/c4x/Engineering/CS-224N/asset/slp4.pdf)
	- [From Languages to Information](http://web.stanford.edu/class/cs124/lec/languagemodeling2016.pdf)


#### A New Paradigm
- So far, we have identified the problems of speech recognition and provided a traditional ASR solution using feature extraction, HMMs and language models
- These systems have gotten better and better since their introduction in the 1980's
- Deep Neural Networks have become the go to solution for all kinds of large probabilistic problems including speech recognition
- Recurrent Neural Networks (RNNs) can be used because they have temporal memory, an important characteristic for training and decoding speech


#### Deep Neural Networks
- If HMMs work, why do we need a new model?
	- Because of potential
	- Suppose we have all the data and fast computers we need. How far could HMMs take us vs how far could some other model take us?
	- Baidu's Adam Coates showed that additional training of a traditional ASR levels off in accuracy
	- DNNs are unimpressive with small datasets, but they shine as data and model sizes increase
- Model so far:
	- Speech -> Features -> Acoustic Model -> Phonemes -> Words -> Language model -> Text
	- Extract features from the audio speech signal with MFCC
	- Use HMM Acoustic Model to convert to sound units, phonemes, or words
	- Use statistical language models such as N-grams to straighten out language ambiguities and create the final text sequence
- Possible to replace many parts with a multiple layer deep neural network
- Why can they be replaced?
	- In feature extraction, use models based on human sound production and perception to convert a spectrogram into features -> similar intuitively to using CNNs to extract features from image data. Spectrograms are visual representations of speech. So a CNN can find the relevant features for speech in the same way
	- An Acoustic model implemented with HMMs includes transition probabilities to organize time series data -> RNNs can also track time series data through memory. The traditional model also uses HMMs to sequence sound units into words -> RNNs produce probability densities over each time slice. Use a Connectionist Temporal Classification (CTC) layer is used to convert the RNN outputs into words. So we can replace the acoustic portion of the network with a combination of RNNs and CTC layers
	- End-to-end DNN still makes linguistic errors, especially on words that it hasn't seen in enough examples. It should be possible for the system to learn language probabilities from audio data but presently there just isn't enough. N-grams could still be used. Alternatively, a Neural Language Model (NLM) can be trained on massive amounts of available text. Using an NLM layer, the probabilities of spelling and context can be re-scored for the system.


#### Connectionist Tempora




## Project: DNN Speech Recognizer
_Build an Automatic Speech Recognizer using Deep Learning RNN's_


## Links
[What is an Alexa Skill?](https://developer.amazon.com/alexa-skills-kit)
[Alexa Skills Kit (ASK)](https://developer.amazon.com/alexa-skills-kit)
[Amazon Developer Console](https://developer.amazon.com/alexa-skills-kit)
[AWS Lambda](https://aws.amazon.com/lambda/)
[Amazon step-by-step instructions for building Alexa Sample Fact Skill](https://github.com/alexa/skill-sample-nodejs-fact/)
[Alexa Skills Training Courses](bit.ly/asktraining)
- [scipy.fftpack.fft](https://docs.scipy.org/doc/scipy-0.19.0/reference/generated/scipy.fftpack.fft.html)


## Books


