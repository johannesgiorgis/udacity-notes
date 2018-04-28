# AIND - Natural Language Processing Notes
This contains notes and references from Udacity AIND Term 2's Natural Language Processing section.


[Lesson Plan](https://sites.google.com/knowlabs.com/aind-student-resources/lesson-plans/term-2-natural-language-processing)

## Lesson 12: Intro to Natural Language Processing
Find out how Natural Language Processing is being used in the industry, why it is challenging, and learn to design an NLP solution using IBM Watson's cloud-based services.

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

## Lesson 13: Bookworm
Learn how to build a simple question answering agent using IBM Watson.

## Lesson 14: Natural Language Processing
An overview of how to build an end-to-end Natural Language Processing Pipeline.

## Lesson 15: Text Processing
Learn to prepare text obtained from different sources for further processing, by cleaning, normalizing and splitting it into individual words or tokens.

## Lesson 16: Feature Extraction
Transform text using methods like Bag-of-Words, TF-IDF, Word2Vec and GloVe to extract features that you can use in machine learning models.

## Lesson 17: Modeling
A selection of different NLP tasks and how to build models that accomplish them.

## Project: Machine Translation
Apply the skills you've learnt in Natural Language Processing to the challenging and extremely rewarding task of Machine Translation.

## Lesson 19: Embeddings and Word2Vec
Learn about embeddings in neural networks by implementing the word2vec model.

## Lesson 20: Sequence to Sequence
Learn about a specific architecture of RNNs for generating one sequence from another sequence. These RNNs are useful for chatbots, machine translation and more!

## Links
- [Grammars](https://classroom.udacity.com/courses/cs101/lessons/48299949/concepts/487192400923)
- [Part of Speech](http://partofspeech.org/)

