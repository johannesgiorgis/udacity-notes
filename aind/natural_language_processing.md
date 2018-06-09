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
- Processing stage begins with reading text data from one of several sources:
	- plain text file on local machine (simplest source)
	- part of a larger database or table (csv file)
	- online resource - web service or API


#### Cleaning
- Text data, especially from online sources is almost never clean
- What did we achieve?
	- Fetched a single web page, Udacity course catalog
	- Tried a couple of methods to remove HTML tags. Settled on using BeautifulSoup to parse the entire HTML source, find all course summaries and extract the title and description for each course, then saved them all in a list
	- Can consider all of these as 1 document or treat them as separate documents. The latter is useful if you want to group related courses. The problem reduces to document clustering

#### Normalization
- Plain text contains all the variations and bells and whistles of human languages. We have to reduce some of that complexity
- Two most common normalization steps:
	- Case Normalization: Convert every letter to a common case, usually lowercase, so that each word is represented by a unique token - ```text = text.lower()```
	- Remove Punctuation: Depending on your NLP task, you may want to remove special characters - periods, question marks, commas - ```text = re.sub(r"[^a-zA-Z0-9]", " ", text)```


#### Tokenization
- Token is a fancy term for a symbol, usually one that holds some meaning and is not typically split up any further
- In NLP, tokens are individual words
- Tokenization is simply splitting each sentence into a sequence of words. Simplest way to do this is via the `split` method, which returns a list of words. It splits on whitespace character by default (space, tabs...etc)
- Python's built-in functionality allows us to do all the previous operations, but some of these operations are much easier to perform using a library like NLTK - Natural Language ToolKit
- `from nltk.tokenize import word_tokenize
words = word_tokenize(text)
print(words)`
- Sentence Tokenization
- `from nltk.tokenize import sent_tokenize
sentences = sent_tokenize(text)
print(sentences)`
- NLTK provides several tokenizers
	- regular expression base tokenizer to remove punctuation and perform tokenization in a single step
	- tweet tokenizer that is aware of twitter handles, hash tags & emoticons


#### Stop Word Removal
- Stop words are uninformative words - is, our, the, in, at...etc that do not add a lot of meaning to a sentence
- They are typically very comonly occuring words and we may want to remove them to reduce the vocabulary we have to deal with and hence the complexity of later procedures
- e.g. "dogs are the best" -> "dogs best"
- We can still infer the sentence's positive sentiment towards dogs with the stop words removed
- NLTK Stop Words
- `from nltk.corpus import stopwords
print(stopwords.words("english"))`
- To remove stop words: `words = [w for w in words if w not in stopwords.words("english")]`


#### Part-of-Speech Tagging
- Identifying how words are being used in a sentence can help us better understand what is being said. It can also point out relationships between words and recognize cross references
- NLTK makes this easy for us: `from nltk import pos_tag
sentence = word_tokensize("I always lie down to tell a lie.")
pos_tag(sentence)`
- This returns a tag for each word identifying different parts of speech


#### Named Entity Recognition
- Typically noun phrases that refer to some specific object, person or place
- Use `ne_chunk()` function to label named entities in text: `from nltk import pos_tag, ne_chuck from nltk.tokenize import word_tokenize ne_chunk(pos_tag(word_tokensize("Antonio joined Udacity Inc. in California.")))`
- Often used to index and search for news articles, for example, on companies of interest


#### Stemming and Lemmatization
- To further simplify text data, we use several ways to normalize different variations and modifications of words
- Stemming is the process of reducing a word to its stem or root form. e.g. branching, branched, branches -> branch because they convey the same concept. This reduces complexity while retaining the essence of meaning that is carried by words
- Stemming meant to be a fast and crude operation carried out by applying very simple search and replace style rules
- NLTK has several different stemmers
	- PorterStemmer: `from nltk.stem.porter import PorterStemmer
	stemmed = [PorterStemmer().stem(w) for w in words]`
	- SnowballStemmer
	- Other language specific stemmers
- Lemmatization is another technique used to reduce words to a normalized form, which uses a dictionary to map different variants of a word back to its root. With this approached, we are able to reduce non-trivial inflections such as is, was, were back to the root 'be'
- Default lemmatizer in NLTK uses the Wordnet database to reduce words to the root form
- A Lemmatizer needs to know or make an assumption about the part of speech for each word it's trying to transform
- Stemming sometimes results in stems that are not complete words in English
- Lemmatization is similar to stemming with one difference, the final form is also a meaningful word
- Stemming does not need a dictionary like lemmatization does, so stemming maybe a less memory intensive option


#### Summary
- Covered a number of text processing steps
- Summarize a typical workflow looks like
	- Start with plain text sentence
	- Normalize it by converting to lowercase and removing punctuation
	- Split it up into words using a tokenizer
	- Remove stop words to reduce the vocabulary you have to deal with
	- Depending on application, you may choose to apply a combination of stemming and lemmatization to reduce words to the root of stem form. It is common to apply both - lemmatization first, and then stemming
- This procedure converts a natural language sentence into a sequence of normalized tokens which you can use for further analysis

## Lesson 16: Feature Extraction
_Transform text using methods like Bag-of-Words, TF-IDF, Word2Vec and GloVe to extract features that you can use in machine learning models._

#### Feature Extraction
- Once we have our text ready in a clean and normalized form, we need to transform it into features that can be used for modeling
	- e.g. treating each document as a bag of words allows us to compute some simple statistics that characterize it
	- these statistics can be improved by assigning appropriate weights to words using a TF-IDF Scheme, allowing for more accurate comparison between words
	- certain applications require numerical representations of individual words - use word embeddings, a very efficient and powerful method
- learn all these techniques used to extract relevant features from text data


#### Bag of Words
- treats each document as an un-ordered collection, or bag of words
- a document is the unit of text you want to analyze
	- e.g. compare essays submitted by students to check for plagiarism, each essay would be a document
	- analyze the sentiment conveyed by tweets, each tweet would be a document
- To obtain a bag of words from a piece of raw text, you need to simply apply appropriate text processing steps: cleaning, normalizing, splitting into words, stemming, lemmatization...etc. Then treat the resulting tokens as an un-ordered collection or set. So each document in your data set will produce a set of words


#### TF-IDF
- A limitation of the bag-of-words approach is it treats every word as being equally important
	- e.g. when looking at financial documents, cost or price may be a pretty common term
- we can compensate for this by counting the number of documents in which each word occurs (document frequency) and dividing the term frequencies by the document frequency of that term. Gives a metric that is proportional to the frequency of occurence of a term in a document, but inversely proportional to the number of documents it appears in. It highlights the words that are more unique to a document, thus better for characterizing it
- TF-IDF transform is the product of two weights: term frequency and inverse document frequency
- most commonly used form of TF-IDF defines:
	- term frequency as the raw count of a term t, in a document d, divided by the total number of terms in d
	- inverse document frequency as the logarithm of the total number of documents in the collection, d divided by the number of documents where t is present
- Several variations exist that try to normalize or smooth the resulting values or prevent edge cases such as divide by zero errors
- Overall, TF-IDF is an innovative approach to assigning weights to words that signify their relevance in documents

#### One-Hot Encoding
- So far, we've looked at representations that tried to characterize an entire document or collection of words as one unit. The kinds of inferences we can make are at the document level: mixture of topics in the document, documents similarity, documents sentiment...etc.
- For a deeper analysis of text, we need to come up with a numerical representation for each word. One-Hot Encoding treats each word like a class, assign it a vector that has one in a single pre-determined position for that word and zero for everywhere else. It is similar to the bag-of-words idea, except we keep a single word in each bag and build a vector for it


#### Word Embeddings
- One-hot encoding breaks down when we have a large vocabulary to deal with as the size of our word representation grows with the number of words. We need to control the size of our word representation by limiting it to a fixed-size vector. 
- We want to find an embedding for each word in some vector space and we want to exhibit some desired properties
	- e.g. if two words are similar in meaning, they should be closer to each other compared to words that are not. if two words have a similar difference in their meanings (man vs woman, king vs queen), they should be approximately equally seperated in the embedded space 
- We can use such a representation for a variety of purposes:
	- finding synonyms and analogies
	- identifying concepts around which words are clustered
	- classifying words as positive, negative, neutral
- By combining word vectors, we can come up with another way of representing documents as well


#### Word2Vec
- Most popular examples of word embeddings used in practice
- It transforms words to vectors
- Core idea of Word2Vec is:
	- a model that is able to predict a given word, given neighboring words, or vice versa, predict neighboring words for a given word is likely to capture the contextual meanings of words very well
	- these are 2 flavors of Word2Vec models: 
		- one where you are given neighboring words called _Continuous Bag of Words (CBoW)_
		- where you are given the middle word call _Continuous Skip-gram_
- In the Skip-gram Model, pick any word in the sentence, convert it to a one-hot encoded vector and feed it into a neural network/probabilistic model designed to predict a few surrounding words, its context. Using an appropriate loss function, optimize the weights or parameters of the model and repeat this until it learns to predict context words as best as it can. Now, take an intermediate representation like a hidden layer in a neural network. The outputs of that layer for a given word become the corresponding word vector.
- The Continuous Bag of Words variation also uses a similar strategy
- Word2Vec Properties
	- This approach yields a robust representation of words as the meaning of each word is distributed throughout the vector
	- The size of the word vector is up to the user depending on how they want to tune performance vs complexity. It remains constant regardless of number of words trained on it unlike Bag-of-words where the size grows with the number of unique words
	- Train once, store in a lookup table to be used again
	- Ready to be used in deep learning architectures
		- e.g. can be used as the input vector for recurrent neural nets.
		- possible to use RNNs to learn even better word embeddings
- Some other optimizations are possible that further reduce the model and training complexity:
	- representing the output words using Hierarchical Soft-max
	- computing loss using Sparse Cross Entropy

#### GloVe
- Word2Vec is just one type of word embedding
- Recently other types of promising approaches have been proposed
- GloVe: Global Vectors for Word Representation
	- tried to directly optimize the vector representation of each word just using co-occurrence statistics, unlike Word2Vec which sets up an ancillary prediction task
- Why co-occurrence probabilities?
- Too complicated to write as text
- TO DO: come up with nice summary of GloVe
- TO DO: Read the original paper that introduced GloVe

#### Embeddings for Deep Learning
- Embeddings are fast becoming the de-facto choice for representing words, especially for use in deep neural networks
- Why do these techniques work so well?
	- Distributional hypothesis states that words that occur in the same context tend to have similar meanings
	- When a large collection of sentences is used to learn in embedding, words with common context words tend to get pulled closer and closer together
- How do we capture similarities and differences in the same embedding? e.g. in some context, tea and coffee can be similar, in other contexts not so much
	- By adding another dimension. Words can be close along one dimension (tea and coffee are beverages), but separated along some other dimension (capture the variability among beverages)
	- In a human language, there are many more dimensions along which word meanings can vary
	- The more dimensions you can capture in your word vector, the more expressive that representation will be

#### t-SNE
- Word embeddings need to have high dimensionality in order to capture sufficient variations in natural language, which makes them super hard to visualize
- t-Distributed Stochastic Neighbor Embedding is a dimensionality reduction technique that can map high dimensional vectors to a lower dimensional space
	- When performing the transformation, it tried to maintain relative distances between objects, so that similar ones stay closer together while dissimilar objects stay further apart
	- Great choice for visualizing word embeddings as it preserves the linear substructures and relationships that have been learned by the embedding model
	- Looking at larger vector space allows us to discover meaningful groups of related words
	- Most groupings are very intuitive though it may take time to understand why
	- t-SNE works on other kinds of data, such as images
- This is a very useful tool for better understanding the representation that a network learns and for identifying any bugs or other issues.

## Lesson 17: Modeling
_A selection of different NLP tasks and how to build models that accomplish them._

#### Modeling
- A model represents observations in a form that allows us to understand them better and predict new occurrences
- Learn how to build models that accomplish various NLP tasks
	- Classification models for sentiment analysis and spam detection
	- Topic modeling for grouping related documents
	- Ranking for improving search relevance
	- Machine translation systems for converting text from one language to another
- There are other application areas that you can address by extending or modifying these techniques

#### Language Model
- Captures the distributional statistics of words
- Unigram Model - only concerned with the probability of occurrence of a single word
	- In most basic form, we take each unique word in a corpus, _i_, and count how many times it occurs: `from collections import Counter
words = ['humpy', 'dumpty', ... , 'together', 'again']
counts = Counter(words)`
	- Divide the counts by total number of words to normalize the values to a [0, 1] range. This produces a _prior probability distribution_ over all unique words. It describes some interesting properties of the corpus: e.g. which words are frequent and which ones are rare
	- We can use such a model to generate new text: `random.choices(list(counts.keys()), weights=list(counts.values()), k=10)`
- Bigram model - capture how often two words occur next to each other; the co-occurrence probabilities of pairs of words
	- `bigrams = [(words[i], words[i + 1]) for i in range(len(words) - 1)]`
	- Every cell _(i, j)_ represents the odds that whenever word _i_ occurs, it is immediately followed by word _j_.
	- Normalize the counts by dividing each element with the respective row sum to obtain a _conditional probability_, _P(j|i)_, a Bigram Model
	- We use _P(i)_ to generate the first word in the sequence, afterwards use _P(j|i)_ to generate every consequent word
- Use a sparser representation for Bigram Models:
	- `{
    'humpty': {'dumpty': 2, 'together': 1},
    'dumpty': {'had': 1, 'sat': 1},
    'all': {'the': 2},
    ...
}`
- This idea can be generalized to any number of words _n_. e.g. **Trigram Model, P(k|i, j)** is the probability of word _k_ occurring immediately after _i-j_ sequence.
	- Slightly different than words occurring within the vicinity of other words
	- There is an explicit order enforced: _i_ followed by _j_ followed by _k_
	- This gives much more realistic sequences at the cost of added complexity
- Language Models have a wide range of applications
	- In speech recognition systems, words that sound similar (homophones) can be distinguished more easily by computing the likelihood of observing candidate sequence of words
	- Distinguish between writings from different authors, or different historical periods - studying the evolution of language over time

#### Sentiment Analysis
- One of the most popular uses of Natural Language Processing
- An important tool for many reasons:
	- Understanding customer sentiment around a company for making investment decisions
	- Getting a feedback signal for social media and advertising campaigns
	- As a quantitative measure for book and movie reviews
- Either a classification or regression problem depending on whether you want to identify specific emotional categories or labels (positive vs. negative) or a real number that captures a more fine-grained sentiment value
- Pipeline:
	- Start with a given corpus, e.g. movie reviews
	- Process each review as an individual document
	- Extract a set of features that represent it
	- Representation can be a direct document representation - Bag-of-Words, TF-IDF - or a sequence of word vectors combined together
	- Depends on what model you want. e.g. use Bag-of-Words if you want to use an SVM to predict sentiment labels; use word vectors if you want to apply an RNN
	- Pick an appropriate loss function to train your model.
		- e.g. categorical cross entropy for classification
		- mean squared error for regression
- Challenges:
	- Consider these two brief movie reviews:
		- _"I expected this movie to be much better."_
		- _"This movie was much better than I expected."_
	- A typical bag-of-words representation will perceive these two to be almost the same
	- If you allow words to be interpreted in a sequence, then ordering differences can help a model learn the distinctions between different sentiments
	- RNN-based models have proved to be very successful at this task, as they treat text like a sequence, and incorporate information over time
- Lab: Sentiment Analysis
	- From the [Github AIND-NLP repository](https://github.com/udacity/AIND-NLP), work on the _sentiment_analysis.ipynb_ notebook

#### Topic Modeling
- Unsupervised learning problem, where there might not be any clear label or category that applies to each document; instead each document is treated as a mixture of various topics
- Estimating these mixtures and simultaneously identifying the underlying topics is the goal of topic modeling
- Bag-of-Words: Graphical Model
	- All you can observe in documents are the words that make them up
	- bag-of-words model represents the relationship between a set of document objects and a set of word objects
	- For any given document **d** and observe time **t**, it helps answer the question "how likely is it that **d** generated **t**"
	- Q: At most how many parameters are needed to define a Bag-of-Word model for 500 documents and 1000 unique words?
	- A: 500,000. Number of documents times the number of unique words
	- Hint: The model needs to capture the relationship between each <document, word> pair. How many such pairs are there?
- Latent Variables
	- Want to add the notion of a small set of topics (_latent variables_) that actually drive the generation of words in each document
	- Any document is considered to have an underlying mixture of topics associated with it
	- A topic is considered a mixture of words that it is likely to generate
	- Two sets of parameters or probability distributions we need to compute:
		- probability of topic **z** given document **d**
		- probability of term **t** given topic **z**
	- This approach helps decompose the document-term matrix, which can get very large if there are a lot of documents with a lot of words, into two much smaller matrices
	- This technique for topic modeling is called **Latent Dirichlet Allocation (LDA)**
	- Q: How many parameters are needed in an LDA topic model for 500 documents and 1000 unique words, assuming 10 underlying topics?
	- A: Underlying topics greatly reduce the model complexity. The total number of parameters include one for each <document, topic> pair, and one for each <topic, word> pair. If number of documents is |D|, number of topics is |Z| and number of unique words (vocabulary size) is |V|, then the total number of parameters = |D| x |Z| + |Z| x |V| = 500 x 10 + 10 x 1000 = 5000 + 10000 = 15,000. This is an example of matrix factorization.
	- Hint: Identifying the underlying topics can help break down the single large set of parameters into two sets: One that captures the relationship between documents and topics, and another that relates topics to words. Try to count the total number of words by summing up these two parameter sets.
- Prior Probabilities
	- Have only been considering conditional probabilities, but not _joint probability_ of a word and a document or that of a word and topic
	- Need to define some prior probabilities to compute the joint probability. Assume that the documents and topics are drawn from _Dirichlet distributions_
	- Dirichlet Distribution can be thought of as a probability distribution over _mixtures_ in the context of topic modeling
	- 


#### Search and Ranking
- Searching online has become such a common activity. Search itself is not typically considered an AI or machine learning problem, but you can treat it as one
- Looking to perform a query against a set of documents, pull out the ones that seem to match, and then rank them using some relevance criteria
- In a sense, can be considered a regression problem, where the input is a _<query, document>_ pair, and the output is a real number value indicating the _relevance_
- But there is more to it - for a particular query, the absolute relevance value don't matter as much as the values relative to each other for a set of results returned. Therefore, searching and ranking may require a different kind of target or loss function. One possibility is to use a top-n approach, i.e. whether the intended document was present in the top-n, say top 10 results or not.

#### Machine Translation
- Can be thought of as a sequence to sequence learning problem
	- One sequence in the source language coming in and another sequence in the target language coming out
	- Very hard problem
	- Recent advances like Recurrent Neural Networks have shown a lot of improvement
	- A typical approach is to use a recurrent layer to encode the meaning of the sentence by processing the words in a sequence, and then either use a dense or fully-connected layer to produce the output, or use another decoding layer
- Neural Net Architecture for Machine Translation
	- Let's develop a basic neural network architecture for machine translation
- Input Representation
	- Instead of a single word vector or document vector as input, we need to represent each sentence in the source language as a sequence of word vectors. So we convert each word or token into a one-hot encoded vector and stack those vectors into a matrix - this becomes our input to the neural network
	- One common approach to dealing with sequences of different lengths is to take the sequence of maximum length in your corpus and pad each sequence with a special token to make them all the same length
- Basic RNN Architecture
	- Once we have the sequence of word vectors, we can feed back in one at a time to the neural network
	- Embedding Layer
		- Typically first layer of the network
		- Helps enhance the representation of the word
		- Produces a more compact word vector that is then fed into one or more recurrent layers
	- Recurrent Layer(s)
		- Where the magic happens!
		- Help incorporate information from across the sequence, allowing each output word to be affected by potentially any previous input word
		- _Note: you can skip the embedding step, which will reduce the complexity of the model and make it easier to train, but the quality of translation may suffer as one-hot encoded vectors cannot exploit similarities and differences between words_
	- Dense Layer(s)
		- Recurrent layer(s) output is fed into one or more fully connected dense layers that produce softmax output, which can be interpreted as one-hot encoded words in the target language
		- As each word is passed in as input, its corresponding translation is obtained from the final output. The output words are then collected in a sequence to produce the complete translation of the input sentence
		- _Note: For efficient processing we would like to capture the output in a matrix of fixed size, which requires output sequences of the same length_
- Recurrent Layer: Internals
	- The input word vector _x<sub>t</sub>_ is first multiplied by the weight matrix: _W<sub>x</sub>_
	- Then bias values are added to produce our first intermediate result: x<sub>t</sub> * W<sub>x</sub> + b
	- Meanwhile, the state vector from the previous time step _h<sub>t-1</sub>_ is multiplied with another weight matrix _W<sub>h</sub>_ to produce our second intermediate result: _h<sub>t-1</sub> * W<sub>h</sub>_
	- These two are then added together, and passed through an activation function such as ReLU, sigmoid or tanh to produce the state for the current time step: _h<sub>t</sub>_
	- This state vector is passed on as input to the next fully-connected layer, that applies another weight matrix, bias and activation to produce the output: _y<sub>h</sub>_
	- The key thing is the RNN's state _h<sub>t</sub>_ is used to produce the output _y<sub>h</sub>_ as well as looped back to produce the next state
- In summary, a recurrent layer computes the current state as:
	- _h<sub>t</sub>_ = f(x<sub>t</sub> * W<sub>x</sub> + h<sub>t-1</sub> * W<sub>h</sub> + b)
- Understanding how the parameters interact with the input and state vectors is very important for correctly designing and debugging RNNs, and deep neural networks in general. When in doubt, write down the mathematical expression and verify that the sizes are consistent with the matrix operations	
- Unrolling an RNN
	- Each copy of the network you see represents its state at the respective time step. At any time _t_, the recurrent layer receives input x<sub>t</sub> as well as the state vector from the previous step, _h<sub>t-1</sub>_. This process is continued till the entire input is exhausted.
	- The main drawback of such a simple model is that we are trying to read the corresponding output for each input word immediately. This would only work in situations where the source and target language have an almost one-to-one mapping between words.
- Encoder-Decoder Architecture
	- Need to let the network learn an internal representation of the entire input sentence, then start generating the output translation. Need 2 different networks to achieve this.
	- Encoder: accepts the source sentence, one word at a time, and captures its overall meaning in a single vector. This is the state vector at the last time step. It isn't used to produce any outputs
	- Decoder: interprets the final sentence vector and expands it into the corresponding sentence in the target language, one word at a time.
	- The first time step for the decoder network is special. It is fed in the final sentence vector from the encoder _h<sub>t</sub>_, and given a sentinel input to kickstart the process. The recurrent portion of the network produces a state vector _c<sub>0</sub>_, and with that the fully-connected portion produces the first output word in the target language, y<sub>0</sub>.
	- At each subsequent time step _t_, the decoder network uses its own previous state _c<sub>t-1</sub>_ as well as its own previous output _y<sub>t-1</sub>_, in order to produce the current output, _y<sub>t</sub>_.
	- Process continued for fixed number of iterations with the network starting to produce special padding symbols after all meaningful words have been generated. Alternately, the network could be trained to output a stop symbol, such as a period (.), to indicate that the translation is complete.
- Encoder-decoder design is very popular for several sequence to sequence tasks, not just Machine Translation:
	- use different kinds of RNN units (LSTMS, GRUs) instead of vanilla RNN units allowing network to better analyze the input sequence at the cost of additional model complexity
	- Explore how many recurrent layers to use. Each layer effectively incorporates information from the input sequence, producing a compact state vector at each time step. Additional layers can essentially incorporate information across these state vectors.
	- More innovative approaches include adding a backward encoder (bidirectional encoder-decoder model), feeding in the sentence vector to each time step of the decoder (attention mechanism)...etc

#### NLP Resources
- Sebastian Ruder, 2017. [Deep Learning for NLP Best Practices](http://ruder.io/deep-learning-nlp-best-practices/): Talks about several cutting-edge mechanisms being developed for NLP and how to best apply them, such as: Multi-Task Learning, Attention, Hyperparameter Optimization, Ensembling.

- Chris Manning and Richard Socher, 2017. [Natural Language Processing with Deep Learning](http://web.stanford.edu/class/cs224n/) (course).
Great for learning about: Advanced Word Embeddings, Dependency Parsing, Coreference Resolution, Gated Recurrent Units.

- Dan Jurafsky and James H. [Speech and Language Processing](http://www.cs.colorado.edu/~martin/slp2.html), 2nd ed. [[3rd ed. drafts](https://web.stanford.edu/~jurafsky/slp3/) | [2017 course](http://web.stanford.edu/class/cs124/)]
A comprehensive study of language processing and the related fields of speech recognition and synthesis. Covers in depth: Statistical Parsing, Information Retrieval, Question-Answering, Summarization.

#### Summary


## Project: Machine Translation
_Apply the skills you've learnt in Natural Language Processing to the challenging and extremely rewarding task of Machine Translation._


## Lesson 19: Embeddings and Word2Vec
_Learn about embeddings in neural networks by implementing the Word2Vec model._

	
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
- [Conversation Service|Watson Assistant](https://console.bluemix.net/docs/services/conversation/index.html#about)
- [Bluemix Console](https://console.bluemix.net/?cm_mmc=dw-_-edtech-_-udacity-_-ai)
- [IBM Bluemix Documentation](https://console.ng.bluemix.net/docs/?cm_mmc=dw-_-edtech-_-udacity-_-ai)
- [Deploying Apps](https://console.bluemix.net/docs/manageapps/depapps.html#deployingapps?cm_mmc=dw-_-edtech-_-udacity-_-ai)
- [Kingfisher Wikipedia](https://en.wikipedia.org/wiki/Kingfisher)
- [WordNet Visualization Tool](http://mateogianolio.com/wordnet-visualization/)
- [Ancient Egypt Writing Wikipedia Page](https://en.wikipedia.org/wiki/Ancient_Egypt#Writing)
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/News+Aggregator)
- [Pandas: Working with Text Data](https://pandas.pydata.org/pandas-docs/stable/text.html)
- [Quote of the Day API](http://quotes.rest/)
- [Udacity Text Processing Notebook](https://github.com/udacity/AIND-NLP/blob/master/text_processing.ipynb)
- [Regular Expressions in Python](https://docs.python.org/3/library/re.html)
- [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)
- [NLTK Tokenize Package](http://www.nltk.org/api/nltk.tokenize.html)
- [Python re library documentation](https://docs.python.org/3.5/library/re.html)
- [Rule-based Machine Translation Wiki](https://en.wikipedia.org/wiki/Rule-based_machine_translation)
- [Statistical Machine Translation Wiki](https://en.wikipedia.org/wiki/Statistical_machine_translation)
- [Example-based Machine Translation Wiki](https://en.wikipedia.org/wiki/Example-based_machine_translation)

## Books
- The Master Algorithm
- Humans are Underrated: What High Achievers Know That Brilliant Machines Never Will

