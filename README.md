# Textual-Entailment-via-Tensorflow


An LSTM solution for detecting textual entailment given two sentences, based on SemEval 2014 [Task 1](http://alt.qcri.org/semeval2014/task1/). 


### Prerequisites

This was built with

* [Tensorflow](https://www.tensorflow.org/install/) - Machine Learning framework
* [spaCy](https://spacy.io/) NLP framework
* [Jupyter](https://jupyter.org/)  
* [TQDM](https://pypi.org/project/tqdm/) - needed for progress bars 
* [Matplotlib](https://matplotlib.org/) 

### Installation

If all the prerequisites are up and running on your computer, all you need to do is download GloVe pretrained vectors, as available [here](http://nlp.stanford.edu/data/glove.6B.zip).
Unzip the file and drop the glove.6B.50d.txt in the project folder. You can change it to any of the other files in the zip, the code just needs to be adjusted for that in the second cell:



```
GloVe_vectors_file = "glove.6B.50d.txt"
```


spaCy english model can be installed by typing the following command in your terminal:

```
python -m spacy download en
```

If all the dependencies are installed and all the data is in the working directory, run 

```
jupyter notebook
```

and select entailment_via_tf.ipynb where you can start running the notebook as you would with any other notebook application (Shift+Enter to run a cell).


##  How it works

To detect entailment between two sentences, we use word vectors so that it is easy to word similarity, which essentially is calculating euclidean distance between these points in a high-dimensional space.
Sentences are turned into sequences of vectors that are fed into the RNN (made up of lstm cells) which has three output nodes, each representing the likelihood of the sentences' entailment. The first node corresponds to positive entailment, the second to neutral(unrelated sentences), and the third to a contradiction between the two.
Then the argmax of the three output nodes is taken, and that is the prediction made by the neural network which is checked with the ground truth labels to calculate accuracy.

### Testing 

Once the model is trained, tests are provided to check the overall accuracy of the system. As of now the results sit at around 60%.

### Future developments
* Additional preprocessing 
* Data augumentation (given the small size of the dataset)

## Acknowledgments

* Guided by [Steven Hewitt](https://github.com/Steven-Hewitt) through his guide over [here](https://www.oreilly.com/learning/textual-entailment-with-tensorflow) !!!
* This was done in the context of CS4040: Natural Language Processing course in the University of Aberdeen.
