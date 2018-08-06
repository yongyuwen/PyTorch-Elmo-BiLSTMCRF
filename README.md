# PyTorch-Elmo-BiLSTMCRF

PyTorch Implementation of the BiLSTM-CRF model as described in https://guillaumegenthial.github.io/. 

This model builds upon that by adding including ELMO embeddings as a feature representation option. 
(For more detail about ELMo, please see the publication ["Deep contextualized word representations"](http://arxiv.org/abs/1802.05365))

For the Keras implementation (without ELMO) please refer to this [link](https://github.com/yongyuwen/sequence-tagging-ner).

## Usage
1.	**Requirements**:  
    a.	Packages: Anaconda, Pytorch, AllenNLP (if on linux and using elmo)  
    b.	Data: Train, valid and test datasets in CoNLL 2003 NER format.  
    c.	Glove 300B embeddings (If not using Elmo) 
    
2.	**Configure Settings**:  
    a.	Change settings in model/config.py  
    b.	Main settings to change: File directories, model hyperparameters etc.  
    
3.	**Build Data**:  
    a.	Run build_data.py  
        i.	Builds embedding dictionary, text file of words, chars tags, as well as idx to word and idx to char mapping for the model to read  
        
4.	**Train Model**:  
    a.	Run train.py  
    
5.	**Test Model**:  
    a.	Run test.py  
    b.	Evaluates on test set. Also accepts other arguments to predict on custom string
