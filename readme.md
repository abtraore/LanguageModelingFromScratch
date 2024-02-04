# Language Modeling From Scratch (LMFS) [Work in progress]

The repository contains a series of algorithms for language modeling (LM). I tried to use the LM to generate few malian family names.

## BiGRAM

For this experiment I arranged the the character as bigrams. The objective is to generate the next charcter based on the first.

### Frequency

I computed the probality distribution by using the bigrams word frequency. More information in `bigram_count.py`. Here is the the frequency table:

<img src="./assets/frequency.png" width="50%">

Some generated names:

- zénema
- kïwicicio

### Perceptron

Instead of manually computing the probablity distrubtion here try to minimize the negative loss likelihood using the gradient descent algorithm and a perceptron. More information in `bigram_perceptron.py`

Some generated names:

- yogarackoue
- mikiafanisa
- tame
- koma
- koko
