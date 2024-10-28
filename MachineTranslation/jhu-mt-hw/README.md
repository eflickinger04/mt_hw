#  Assignments for [EN 601.468/668 Machine Translation](http://mt-class.org/jhu/)

This is the repo for homework assignment for JHU Machine Translation class.

Homework 2-3 are originally from https://github.com/alopez/en600.468

Homework 4-5 are designed by Huda Khayrallah and Brian Thompson

This code runs a neural translation model using the test, train, and development data in the \data folder. 

To run the model with default parameters, use the commadnd 'python seq2seq.py'. 

The model uses and attention mechanism with the encoder and decoder along with a custom lstm to translate french to english sentences. The output contains the translations, BLEU score, and a visualization of the attention mechanism in the form of a heat map. 