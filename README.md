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

# Training on Google-Colab 
To train a transformer encoder-decoder model, use 
* `!git clone <github_access_token_>@github.com/DwaraknathT/GSoC-19.git `- You can get an access token in the developer settings 


* `!cd GSoC-19 `
  `!pip install -r GSoC-19/requirements.txt` - To install all the required packages 
  
  
* `!python GSoC-19/train.py -data GSoC-19/data/baseline -save_model baseline -world_size 1 -gpu_ranks 0 -encoder_type transformer -decoder_type transformer -heads 8 -layers 8 -rnn_size 1024 -word_vec_size 1024 -optim adam -position_encoding -dropout 0.5 -batch_size 64 -train_steps 100000 -learning_rate 0.001 -save_checkpoint_steps 5000`-Trains a transformer model 
