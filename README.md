# GSoC-19
A Multilingual RDF verbalizer for DBpedia- Google Summer of Code- 2019. 

# Implementation 
This Implementation is an extension of the OpenNMT-py code. 
Useful modules are also taken from Graph-2-text , especially 
the preprocessing modules. 

# Abstract: 
A Multilingual Neural model that takes RDF triples as input, 
and outputs a natural language description. 

Ex - 

Input:  <dwarak | Birthplace | Chennai> 
<Chennai | capital | TamilNadu> 

Output:   Dwarak was born in Chennai, capital of TamilNadu. 

# Idea:
Encoder- Decoder seq2seq architecture, with Graph Attention Network 
encoder and Transformer decoder. 
