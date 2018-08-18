Towards Capsule Networks (CapsNets): A Brief Overview of Neural Networks and Deep Learning

Date: July 31, 2018




Prepared By:

Sweta Karlekar, University of North Carolina, Chapel Hill
Abhishek Bhargava, Carnegie Mellon University
Melvin Dedicatoria, The MITRE Corporation
Shawn Na, The MITRE Corporation

Table of Contents

Table of Figures	1
Introduction	3
Brief History of Neural Networks	4
Single-Layer Neural Networks: Perceptrons	5
Organization	6
How do Neural Networks Learn?	7
Backpropagation	7
Loss Function	8
Gradient Descent	8
Types of Learning	9
Supervised Learning	9
Unsupervised Learning	10
Convolutional Neural Networks	11
Motivation	11
Convolution Function	12
Organization	12
Hinton’s Capsule Networks	15
Motivation	15
Organization	16
Future Work	19
References	0

#Introduction

As a precursor to deep learning and machine learning, rule-based systems were the first form of feature-processing programs. Rule-based systems are entirely deterministic and require users to hand-design the whole program using experts in each respective task. For example, a simple rule-based system for classifying if language is considered polite can search for more than two instances of the words “please” and “thank you”. Of course, much larger and more complex rules can be built, and there are many rule-based systems in use online and in various applications today. However, there is no “learning” taking place in rule-based systems. 

The next iteration of Artificial Intelligence (AI) systems took form as classical machine learning. In classical machine learning, humans are required to hand-design each feature, or define the characteristics the system should look for. However, the system learns to map the features to outputs and optimize how much it pays attention to each feature to get the best fit for the data. Returning to our previous example, we might define politeness-features as number of times “please” and “thank you” is said, the length of a greeting, and the presence of good-byes. We don’t know how important each feature is in relation to each other or the sentence, nor do we know what kinds of cutoffs we need (i.e. How do we know the proper length of a greeting to indicate politeness?). However, classical machine learning can dynamically learn the importance, or weighting, of each hand-crafted feature towards making a final decision. For example, the system may learn the number of times “please” and “thank you” occur in a sentence are more important than the length of its greeting. 

In deep learning, the system can automatically discover features at various levels of composition and map these features to the output. Deep learning can drastically reduce pre-processing time as human experts are not needed to hand-craft features. Moreover, due to the system’s ability to discover features at various levels of composition, neural networks can learn higher-level abstractions of patterns which may be lost to other forms of AI systems. Neural networks are defined as a large and varied set of algorithms and models that are loosely designed to mimic the human brain and human intelligence to perform tasks that require pattern recognition. Unlike standard computing programs, neural networks are not necessarily deterministic nor sequential, and respond dynamically to their inputs. Moreover, the information gained by the network from the input is not stored in external memory, but rather in the network itself, much like the human brain. In the example of detecting politeness in language, running neural networks over the language can produce abstract features such as deference and gratitude, as well as lower-level features such as the presence of various forms of punctuation and indefinite pronouns (for a real example, see https://arxiv.org/pdf/1610.02683.pdf). 

For a multitude of tasks, deep learning has been shown to achieve new state-of-the-art standards, outperforming classical machine learning methods. However, it is important to note that as components of AI systems become automatic, errors are introduced earlier in the pipeline and thus can propagate through the system. Humans have less control over the final classification, which can lead to a higher margin of error for some domains. Furthermore, deep learning has received criticism because the user does not know which features the network is finding (i.e. neural networks are highly “black-boxed”). Even with visualization techniques to better understand learned features, deep learning should not be applied to tasks that are very error-sensitive or require concrete interpretability. For these tasks, rule-based or classical machine learning approaches are the most suitable and allow humans to have complete control over the features that are detected. 

#Brief History of Neural Networks 

1943: McCulloch and Pitts introduce the first neural networking computing model. 
1957: Rosenblatt introduce the Perceptron, a single-layer neural network. 
1960: Henry J. Kelly introduces control theory, leading to develop of basic continuous backpropagation model. This method will not gain widespread recognition in the community until 1986. 
1965: Ivakhnenko & Lapa developed group method of data handling (GMDH), a group of algorithms that provide the foundation of deep learning by detailing how datasets can lead to optimized models, and demonstrate its application to shallow neural networks. 
1971: Ivakhnenko successfully demonstrates the deep learning process by creating an 8-layer deep network that serves as a computer identification system called Alpha. 
1974-1980: First AI Winter occurs, a period of reduced research funding and interest. 
1979-1980: Kunihiko Fukushima creates the Neocognitron, a neural network that recognizes visual patterns. His work leads to the development of the first convolutional neural networks (CNNs). 
1982: John Hopfield creates Hopfield networks, a type of recurrent neural network (RNN) that serve as a content-addressable memory system. Hopfield networks are still a commonly used deep learning tool today. 
1986: Rumelhart, Hinton, and Williams describe backpropagation in greater detail and show how it can improve shape recognition, word prediction, and more in their paper “Learning Representations by Back-propagating Errors”. This method was shown to vastly improve existing neural networks. 
1987-1993: Second AI Winter occurs. 
1989: Yann LeCun combines the CNN with recent research done on backpropagation to read handwritten digits. This system was eventually used by NCR and other companies to read zip codes and process cashed checks in the late 90s and early 2000s. 
1997: IBM’s Deep Blue beets chess grandmaster Garry Kasparov. 
1998: Yann LeCun proposes stochastic gradient descent algorithm, and gradient-based learning quickly takes hold as the preferred and successful approach to deep learning.  
2009: Fei-Fei Li launches ImageNet, a database of more than 14 million labeled images for researchers, educators, and students. This launches a new interest in data-driven learning and offers more accessible data at the university level for improvements in deep learning and computer vision. 
2011: Alex Krizhevsky improves on LeNet (the fifth iteration) by strengthening speed and dropout using ReLU (rectified linear units), kicking off renewed interest in CNNs. 
2011: IBM’s Watson wins against Ken Jennings and Brad Rutter on Jeopardy. 
2014: DeepFace created by Facebook to identify faces with 97.35% accuracy, demonstrating that deep learning accuracies can rival that of human accuracy (97.5%). 
2016: Google’s AlphaGo beats Lee Sedol, a top-ranked player, in a Go five-game match (also known as Google DeepMind Challenge match). 

