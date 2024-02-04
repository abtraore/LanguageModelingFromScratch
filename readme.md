# Language Modeling From Scratch (LMFS) [Work in progress]

The repository contains a series of algorithms for language modeling (LM). I tried to use the LM to generate few malian family names.

## BiGRAM and word frequency.

I arranged the the character as bigrams, the objective is to generate the next charcter based on the first. I computed the probality distribution by using the the words frequency. More information in `bigram_count.py`. Here is the the frequency table:

<img src="./assets/frequency.png" width="50%">

Some generated names from the model:

- zénema.
- kïwicicio.
