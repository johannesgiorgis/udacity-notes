# AIND - Natural Language Processing Notes
_This contains notes and references from Udacity AIND Term 2's Natural Language Processing section._


[Lesson Plan](https://sites.google.com/knowlabs.com/aind-student-resources/lesson-plans/term-2-natural-language-processing)

## Lesson 12: Intro to Natural Language Processing
_Find out how Natural Language Processing is being used in the industry, why it is challenging, and learn to design an NLP solution using IBM Watson's cloud-based services._

#### NLP Overview
- Various artificial systmes we interact with every day - phones, cars, websites, coffee machines. It is natural to expect them to process and understand human language.
- Learn how to design an intelligent application that uses NLP techniques and deploy it on a scalable platform.


#### Structured Languages
- Human languages lack precisely defined structures
- Mathematics uses a structured language. No ambiguity in `y = 2x + 5`
- Formal logic uses a structured language: `Parent(X, Y) ^ Parent(X, Z) -> Sibling(Y, Z)`
- Scripting and Programming languages are structured languages. SQL Example: `SELECT name, email FROM users WHERE name LIKE 'A%'`


#### Grammar
- Structured languages are easy to parse and understand for computers because they are defined by a strict set of rules or grammar.
- There are standard forms of expressing such grammars and algorithms that can parse properly formed statements to understand exactly what is meant.
- When a statement doesn't match the prescribed grammar, a typical computer doesn't try to guess the meaning, it simply gives up. Such violations of grammatical rules are reported as syntax errors.


#### Unstructured Text
- Languages we use to communicate with each other also have defined grammatical rules.
- In some situations we use simple structured sentences, but human discourse is mostly complex and unstructured.
- What can computers do to make sense of unstructured text?
- Computers can do:
	- Process words and phrases
		- Keywords
		- Parts of Speech
		- Named entities
		- Dates and quantities
	- Parse sentences using above information
		- Statements
		- Questions
		- Instructions
	- Analyze Documents
		- Frequent and rare words
		- Tone and sentiment
		- Document clustering
- Building on top of these ideas, computers can do a lot with unstructured text.
- She works at IBM -> pronoun verb preposition 'named entity'


#### Counting Words
- Counting word frequencies is often used in NLP
- Counting word code quiz


#### Context is Everything
- What is stopping computers from as capable as humans in natural language processing?
- Part of the problem lies in the variability and complexity of our sentences.
- Sentence examples:
	- _I was lured to see this on the promise of a smart, witty slice of old fashioned fun and intrigue - I was conned._
	- _The sofa didn't fit through the door because it was too narrow_
	- _The sofa didn't fit through the door because it was too wide_
- To understand the proper meaning or semantics of the sentence you implicitly applied your knowledge about the physical world, that wide things don't fit through narrow things. You may know this also from experience.
- There are countless other scenarios in which some knowledge or context is indispensable for correctly understanding what is being said.


#### NLP and IBM Watson
- To understand human language is to understand not only the words but the concepts and how they are linked together to create meaning
- The ambiguity of language is what makes NLP a difficult problem for computers to master
- e.g. _This fountain is not drinking water_
- A cognitive system that can tokenize, parse and annotate the phrase will infer that hte fountain is currently not engaged in the act of drinking water. But we know it means we should not drink the water from this fountain.
- Today's systems go beyond the utterances of the words. They understand the relationship of the words, the contexxt surrounding the utterances.
- Why do I need to scan through dozens of AirBNB reviews to find a suitable accomodation when I can better describe what I want in my own words?
	- _I am looking for an apartment with views of the river, facing the Manhattan skyline, but I don't want a noisy and smelly location_
	- The cognitive system should be able to uncover the customer reviews that depict exacty the criteria that I am seeking.
- NLP can help understand a user's intentions and what they like or dislike and process large volumes of text to better infer the context of our conversations and just exactly what we mean by what we say.

#### Applications of NLP
- Example 1: Metrics to measure how effectively coaches were teaching apprentice coaches to become professionals and take on their own
	- Speech to text
	- Tone Analyzer
	- Sentiment Analysis
- Example 2: Watson for Oncology helps physicians quickly identify key information in a patient's medical record, surveys relevant articles and explores treatment options to reduce unwanted variations of care, and give time back to their patients.
	- Found to be 90% correct
- What are some ways in which NLP is finding its way into everyday products and services?
	- one to one versus one to many. For the first time in advertising marketing history, brands have the opportunity to reach consumers on a personal level at the place of play that the consumers dictate
	- small data versus big data. One to one communication gives you personal insights and specifics. It gives you 'snackable' data that is actionable. For the first time, it creates a perfect platform to sequence messages in a natural contextual human way
	- always on versus always perfect.

#### Challenges in NLP
- What are the most important chellenges in NLP that need to be solved?
	- Understanding and maintaining a context throughout the conversation
- What other NLP problems are AI Engineers and researchers actively working on?
	- Arificial Narrow Intelligence (weak AI)
	- Artificial General Intelligence: aware of its surroundings
	- Arificial Super Intelligence: aware of itself
- How are researchers solving these hard challenges and how does deep learning apply to NLP?
	- NLP has it roots in computational linguistics
	- Researches began tackling the problem using rule based systems
	- Due to the inherent complexity and lack of structure in natural language, deep learning approaches are being adopted to solve problems such as semantic analysis and disambiguation
	- A number of associated challenges need to be addressed:
		- How do you transform text into a representation suitable for learning?
		- How do you interpret the output of such a system?
		- Developers playing with Long Short Term Memory (LSTMs)
	- Hot research areas
		- Image Captioning
		- Visual Question Answering
		- combine both image and text
	- Deep learning approaches use a common underlying vector representation for such problems, and they are being applied to them with great success
	- Beyond NLP, these systems may become precursors to more general purpose AI systems


#### NLP Services
- NLP related services in Watson:
	- 2 flagship services: conversation service and discovery service
	- Conversation service is a place where you know what it is that you don't know and so it uncovers the answers for you
	- Discovery service is about what it is that you don't even know that you don't know. It uncovers insights which leads us to more questions and further insights.
	- The discovery service relies heavily on natural language understanding. It includes enrichments such as keyword extraction, doc sentiment, taxonomy, concept expansion...etc. You can even add your own custom models using a SaaS offering called Watson Knowledge Studio.
- NLU is a recent subset of NLP
- The Jeopardy challenge relied heavily on NLP Principles and on brute statistics such as logistical regression
- NLU relies on deep learning technologies
- E.g. To have a meaningful conversation with machines is only possible when we match every word to the correct meaning based on the meanings of the other words in the sentence
- With the conversation service, you are in and out. You have a question, you get an answer.
- With the discovery service, you are a researcher where you spend considerable time gaining insights, which lead to further questions and discoveries
- Quiz - IBM Watson NLP Services
	- Spam Detection -> Natural Language Classifier
	- Customer Satisfaction Monitor -> Tone Analyzer
	- FAQ Bot -> Conversation
	- Query Unstructued Text -> Discovery


#### Getting Started with Watson
- Go to Watson Developer Cloud
- Can launch apps or go to Github repository
- Example of launching an application using IBM Watson services
- Lots of services written in node


#### Deploying a Bluemix Application
- Deploy the application built using IBM Watson services to IBM Bluemix


#### Towards Augmented Intelligence
- AI will not replace jobs as much as it will be augment our work and allow us to focus on the core mission while handling the mundane


#### Project Preview: Bookworm
- Preview of upcoming Bookworm Project

## Lesson 13: Bookworm
_Learn how to build a simple question answering agent using IBM Watson._  
In this project, you will use IBM Watson's NLP Services to create a simple question-answering system. You will first use the Discovery service to pre-process a document collection and extract relevant information. Then you will use the Conversation service to build a natural language interface that can respond to questions.


- [Bookworm Project Github Repository](https://github.com/udacity/AIND-NLP-Bookworm)

## Lesson 14: Natural Language Processing
_An overview of how to build an end-to-end Natural Language Processing Pipeline._

#### NLP and Pipelines
- NLP is one of the fastest growing fields in the world
- It is making its way into a number of products and services we use every day
- Overview of how to design an end to end NLP pipeline
	- start with raw text
	- process it
	- extract relevant features
	- build models to accomplish NLP tasks
- Learn 
	- How these different stages in the pipeline depend on each other
	- How to make design decisions
	- How to choose existing libraries and tools to perform each step

#### How NLP Pipelines Work
- Common NLP pipeline consists of 3 stages:
	- Text Processing
	- Feature Extraction
	- Modeling
- Each stage transforms text in some way and produces a result that the next stage needs.
- Stages:
	- The goal of text processing is take raw input text, clean it, normalize it and convert it into a form that is suitable for feature extraction
	- Next stage needs to extract and produce feature representations that are appropriate for the type of model you're trying to use and the NLP task you're trying to accomplish
- Your workflow may not be entirely linear when building such a pipeline
- Simplified view of NLP. Your application may require additional steps.

#### Text Processing
- Why do we need to process text? Why can we not feed it in directly?
- Think about where we get this text to begin with:
	- Websites are a common source of textual information and fastest growing source
	- PDFs
	- Word Documents
	- Other file formats
	- Raw input from a speech recognition system
	- Book scan using OCR
- Your goal is to extract plain text that is free of any source specific markers or constructs that are not relevant to your task
- Once you have obtained plain text, some further processing may be necessary

#### Feature Extraction
- With clean normalized text, can we feed this into a statistical or a machine learning model? Not quite
- 	Text data is represented on modern computers using an encoding such as ASCII or Unicode that maps every character to a number
- Computers store and transmit these values as binary, zeros and ones
- These numbers have an implicit ordering, but using them in NLP can be misleading
- Individual characters don't carry much meaning at all
- We are concerned about words, but computers don't have a standard representation for words that is capable of capturing the meanings or relationships between words
- Images are represented by a group of pixels in computer memory with each pixel value containing the relative intensity of light at that spot in the image. Pixels serve as a good starting point
- How to do we come up with a similar representation for text data that we can use as features for modeling?
- It depends on what kind of model you are using and what task you're trying to accomplish
	- If you want to use a graph based model to extract insights, you may want to represent words as symbolic nodes with relationships between them like WordNet
	- For statistical models, you need some sort of numerical representation, while keeping the end goal in mind
	- If you're trying to perform a document level task, e.g. spam detection or sentiment analysis, you may want to use a per document representations such as bag-of-words or doc2vec
	- If you want to work with individual words or phrases, e.g. text generation or machine translation, you'll need a word level representation such as word2vec or glove
- There are many ways of representing textual information


#### Modeling
- The final stage in the NLP Pipeline
- Includes:
	- designing a model, usually a statistical or machine learning model
	- fitting its parameters to training data using an optimization procedure
	- then using it to make predictions about unseen data
- Numerical features allow you to utilize any machine learning model
	- Support Vector Machines
	- Decision Trees
	- Neural Networks
	- Any custom model of your choice
	- Combine multiple models to get better performance
- How you use the model is up to you
	- Web Application
	- Mobile App
	- Integrate it with other products, services

## Lesson 15: Text Processing
_Learn to prepare text obtained from different sources for further processing, by cleaning, normalizing and splitting it into individual words or tokens._

#### Text Processing
- Learn how to read text data from different sources and prepare it for feature extraction
	- Begin by cleaning it to remove irrelevant items (e.g. html tags)
	- Normalize text by converting it into all lowercase, removing punctuations and extra spaces
	- Split the text into words or tokens and remove words that are too common, also known as stop words
	- Learn how to identity different parts of speech, named entities annd convert words into canonical form using stemming and lemmatization
- The final result captures the essence of what was being conveyed in a form that is easier to work with


#### Capturing Text Data

#### Cleaning

#### Normalization

#### Tokenization

#### Stop Word Removal

#### Part-of-Speech Tagging

#### Named Entity Recognition

#### Stemming and Lemmatization

#### Summary

## Lesson 16: Feature Extraction
_Transform text using methods like Bag-of-Words, TF-IDF, Word2Vec and GloVe to extract features that you can use in machine learning models._


## Lesson 17: Modeling
_A selection of different NLP tasks and how to build models that accomplish them._


## Project: Machine Translation
_Apply the skills you've learnt in Natural Language Processing to the challenging and extremely rewarding task of Machine Translation._


## Lesson 19: Embeddings and Word2Vec
_Learn about embeddings in neural networks by implementing the word2vec model._

	
## Lesson 20: Sequence to Sequence
_Learn about a specific architecture of RNNs for generating one sequence from another sequence. These RNNs are useful for chatbots, machine translation and more!_

## Links
- [Grammars](https://classroom.udacity.com/courses/cs101/lessons/48299949/concepts/487192400923)
- [Part of Speech](http://partofspeech.org/)
- [Watson Developer Cloud](https://www.ibm.com/watson/developercloud/)
- [Watson Developer Cloud Udacity](https://www.ibm.com/watson/developercloud/?cm_mmc=dw-_-edtech-_-udacity-_-ai) [broken link]
- [Getting Started](https://console.bluemix.net/docs/services/watson/index.html#about)
- [Starter Kits](https://console.bluemix.net/developer/watson/starter-kits)
- [IBM Watson APIs Github Repository](https://github.com/watson-developer-cloud/)
- [Discovery Service](https://console.bluemix.net/docs/services/discovery/index.html#about)
- [Conversation Service|Watson Assitant](https://console.bluemix.net/docs/services/conversation/index.html#about)
- [Bluemix Console](https://console.bluemix.net/?cm_mmc=dw-_-edtech-_-udacity-_-ai)
- [IBM Bluemix Documentation](https://console.ng.bluemix.net/docs/?cm_mmc=dw-_-edtech-_-udacity-_-ai)
- [Deploying apps](https://console.bluemix.net/docs/manageapps/depapps.html#deployingapps?cm_mmc=dw-_-edtech-_-udacity-_-ai)
- [Kingfisher Wikipedia](https://en.wikipedia.org/wiki/Kingfisher)
- [WordNet Visualization Tool](http://mateogianolio.com/wordnet-visualization/)

## Books
- The Master Algorithm
- Humans are Underrated: What High Achievers Know That Brilliant Machines Never Will

