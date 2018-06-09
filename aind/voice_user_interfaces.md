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


## Lesson 24: Introduction to Speech Recognition
_Dive deeper into the exciting field of Speech Recognition, including cutting edge deep learning technologies for Automatic Speech Recognition_


## Project: DNN Speech Recognizer
_Build an Automatic Speech Recognizer using Deep Learning RNN's_


## Links
[What is an Alexa Skill?](https://developer.amazon.com/alexa-skills-kit)
[Alexa Skills Kit (ASK)](https://developer.amazon.com/alexa-skills-kit)
[Amazon Developer Console](https://developer.amazon.com/alexa-skills-kit)
[AWS Lambda](https://aws.amazon.com/lambda/)
[Amazon step-by-step instructions for building Alexa Sample Fact Skill](https://github.com/alexa/skill-sample-nodejs-fact/)
[Alexa Skills Training Courses](bit.ly/asktraining)

## Books


