# Notes

## Sources

<https://arxiv.org/abs/2005.04118>: \[Ribeiro et al.2020\] Marco Tulio Ribeiro, Tongshuang Wu, Carlos Guestrin, and Sameer Singh. 2020. Beyond accuracy: Behavioral testing of NLP models with CheckList. In Proceedings of the 58th Annual Meeting of the
Association for Computational Linguistics, pages 4902–4912, Online, July. Association for Computational Linguistics.

<https://nlp.stanford.edu/pubs/snli_paper.pdf> Samuel R. Bowman, Gabor Angeli, Christopher Potts, and Christopher D. Manning. 2015. A large annotated corpus for learning natural language inference. In Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing (EMNLP).

### Fixing

<https://arxiv.org/pdf/1908.10763.pdf> He He, Sheng Zha, and Haohan Wang. 2019. Unlearn dataset bias in natural language inference by fitting the residual. In Proceedings of the 2nd Workshop on Deep Learning Approaches for Low-Resource NLP (DeepLo 2019), pages 132–142, Hong Kong, China, November. Association for Computational Linguistics.

<https://aclanthology.org/2020.emnlp-main.746/> Swabha Swayamdipta, Roy Schwartz, Nicholas Lourie, Yizhong Wang, Hannaneh Hajishirzi, Noah A. Smith, and Yejin Choi. 2020. Dataset cartography: Mapping and diagnosing datasets with
training dynamics. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing
(EMNLP), pages 9275–9293, Online, November. Association for Computational Linguistics.

<https://arxiv.org/abs/1911.03861> Yadollah Yaghoobzadeh, Soroush Mehri, Remi Tachet des Combes, T. J. Hazen, and
Alessandro Sordoni. 2021. Increasing robustness to spurious correlations using forgettable examples. In Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main
Volume, pages 3319–3332, Online, April. Association for Computational Linguistics.

<https://arxiv.org/abs/2010.03532
> Yixin Nie, Xiang Zhou, and Mohit Bansal. 2020. What can we learn from collective human opinions
on natural language inference data? In Proceedings of the 2020 Conference on Empirical Methods in Natural
Language Processing (EMNLP), pages 9131–9143, Online, November. Association for Computational Linguistics.

## Data Design

### Simiple Negation

#### Entailment

P: `The boy is wearing a green shirt.`  
H: `The boy is not wearing a yellow shirt.`

P: `The boy has on a green shirt.`  
H: `The boy does not have on a yellow shirt.`

#### Neutral

P: `The boy is not wearing a green shirt.`  
H: `The boy is wearing a yellow shirt.`

P: `The boy does not have on a green shirt.`  
H: `The boy has on a yellow shirt.`

#### Contradiction

P: `The boy is wearing a green shirt.`  
H: `The boy is not wearing a green shirt.`

P: `The boy is not wearing a green shirt.`  
H: `The boy is wearing a green shirt.`

P: `The boy has on a green shirt.`  
H: `The boy does not have on a green shirt.`

P: `The boy does not have on a green shirt.`  
H: `The boy has on a green shirt.`

### Slightly Complex Negation

#### Entailment

P: `The boy is wearing a green checkered shirt.`  
H: `The boy is not wearing a yellow checkered shirt.`

P: `The boy has on a green checkered shirt.`  
H: `The boy does not have on a yellow checkered shirt.`

P: `The boy is wearing a green checkered shirt.`  
H: `The boy is not wearing a green striped shirt.`

P: `The boy has on a green plaid shirt.`  
H: `The boy does not have on a green striped shirt.`

#### Neutral

P: `The boy is not wearing a green checkered shirt.`  
H: `The boy is wearing a yellow checkered shirt.`

P: `The boy does not have on a green checkered shirt.`  
H: `The boy has on a yellow checkered shirt.`

P: `The boy is not wearing a green checkered shirt.`  
H: `The boy is wearing a green striped shirt.`

P: `The boy does not have on a green checkered shirt.`  
H: `The boy has on a yellow green striped shirt.`

#### Contradiction

P: `The boy is wearing a green checkered shirt.`  
H: `The boy is not wearing a green checkered shirt.`

P: `The boy is not wearing a green checkered shirt.`  
H: `The boy is wearing a green checkered shirt.`

P: `The boy has on a green checkered shirt.`  
H: `The boy does not have on a green checkered shirt.`

P: `The boy does not have on a green checkered shirt.`  
H: `The boy has on a green checkered shirt.`

### More Complex Negation

#### Entailment

#### Neutral

P: `The boy is not wearing a green checkered shirt.`  
H: `The boy is wearing a green checkered hat.`

P: `The boy does not have on a green checkered shirt.`  
H: `The boy has on a green checkered hat.`

#### Contradiction

#### Structures

##### Persons

[man, woman, boy, girl, father, mother, son, daughter, adult, teenager, child]

##### Colors

['yellow', 'red', 'blue', 'orange', 'purple', 'green', 'black', 'gray', 'white']

##### Uncommon Colors

['cyan', 'magenta', 'chartruese', 'vermillion', 'periwinkle', 'biege', 'taupe']

##### Patterns

['plain', 'striped', 'checkered']

##### Uncommon Patterns

['argyle', 'polkadot', 'tie-dyed']

##### Singular Clothing

['hat', 'shirt', 'tie', 'belt', 'scarf', 'jacket', 'coat', 'scarf']

##### Plural Clothing

['shorts', 'pants', 'gloves', 'shoes', 'socks']

#### Templates

1. Single common color:

    a. Positive  
        `The {person} is wearing a {color} {sing_clothing}.`  
        `The {person} has on a {color} {sing_clothing}.`  
        `The {person} is wearing {color} {plur_clothing}.`  
        `The {person} has on {color} {plur_clothing}.`  
    b. Negative  
        `The {person} is not wearing a {color} {sing_clothing}.`  
        `The {person} does not have on a {color} {sing_clothing}.`  
        `The {person} is not wearing {color} {plur_clothing}.`  
        `The {person} does not have on {color} {plur_clothing}.`  

E:
    a1 -> b1 (change)
    a2 -> b2 (change)
    a3 -> b3 (change)
    a4 -> b4 (change)
N:
    b1 -> a1 (change)
    b2 -> a2 (change)
    b3 -> a3 (change)
    b4 -> a4 (change)
C:
    a1 -> b1 (don't change)
    a2 -> b2 (don't change)
    a3 -> b3 (don't change)
    a4 -> b4 (don't change)
    b1 -> a1 (don't change)
    b2 -> a2 (don't change)
    b3 -> a3 (don't change)
    b4 -> a4 (don't change)

2. Single common pattern:

    a. Positive  
        `The {person} is wearing a {pattern} {sing_clothing}.`  
        `The {person} has on a {pattern} {sing_clothing}.`  
        `The {person} is wearing {pattern} {plur_clothing}.`  
        `The {person} has on {pattern} {plur_clothing}.`  
    b. Negative  
        `The {person} is not wearing a {pattern} {sing_clothing}.`  
        `The {person} does not have on a {pattern} {sing_clothing}.`  
        `The {person} is not wearing {pattern} {plur_clothing}.`  
        `The {person} does not have on {pattern} {plur_clothing}.`  

E:
    a1 -> b1 (change)
    a2 -> b2 (change)
    a3 -> b3 (change)
    a4 -> b4 (change)
N:
    b1 -> a1 (change)
    b2 -> a2 (change)
    b3 -> a3 (change)
    b4 -> a4 (change)
C:
    a1 -> b1 (don't change)
    a2 -> b2 (don't change)
    a3 -> b3 (don't change)
    a4 -> b4 (don't change)
    b1 -> a1 (don't change)
    b2 -> a2 (don't change)
    b3 -> a3 (don't change)
    b4 -> a4 (don't change)

3. Single uncommon color:

    a. Positive  
        `The {person} is wearing a {uncommon_color} {sing_clothing}.`  
        `The {person} has on a {uncommon_color} {sing_clothing}.`  
        `The {person} is wearing {uncommon_color} {plur_clothing}.`  
        `The {person} has on {uncommon_color} {plur_clothing}.`  
    b. Negative  
        `The {person} is not wearing a {uncommon_color} {sing_clothing}.`  
        `The {person} does not have on a {uncommon_color} {sing_clothing}.`  
        `The {person} is not wearing {uncommon_color} {plur_clothing}.`  
        `The {person} does not have on {uncommon_color} {plur_clothing}.`  

E:
    a1 -> b1 (change)
    a2 -> b2 (change)
    a3 -> b3 (change)
    a4 -> b4 (change)
N:
    b1 -> a1 (change)
    b2 -> a2 (change)
    b3 -> a3 (change)
    b4 -> a4 (change)
C:
    a1 -> b1 (don't change)
    a2 -> b2 (don't change)
    a3 -> b3 (don't change)
    a4 -> b4 (don't change)
    b1 -> a1 (don't change)
    b2 -> a2 (don't change)
    b3 -> a3 (don't change)
    b4 -> a4 (don't change)

4. Single uncommon pattern:

    a. Positive  
        `The {person} is wearing a {uncommon_pattern} {sing_clothing}.`  
        `The {person} has on a {uncommon_pattern} {sing_clothing}.`  
        `The {person} is wearing {uncommon_pattern} {plur_clothing}.`  
        `The {person} has on {uncommon_pattern} {plur_clothing}.`  
    b. Negative  
        `The {person} is not wearing a {uncommon_pattern} {sing_clothing}.`  
        `The {person} does not have on a {uncommon_pattern} {sing_clothing}.`  
        `The {person} is not wearing {uncommon_pattern} {plur_clothing}.`  
        `The {person} does not have on {uncommon_pattern} {plur_clothing}.`  

E:
    a1 -> b1 (change)
    a2 -> b2 (change)
    a3 -> b3 (change)
    a4 -> b4 (change)
N:
    b1 -> a1 (change)
    b2 -> a2 (change)
    b3 -> a3 (change)
    b4 -> a4 (change)
C:
    a1 -> b1 (don't change)
    a2 -> b2 (don't change)
    a3 -> b3 (don't change)
    a4 -> b4 (don't change)
    b1 -> a1 (don't change)
    b2 -> a2 (don't change)
    b3 -> a3 (don't change)
    b4 -> a4 (don't change)

5. Common color _and_ common pattern:

    a. Positive  
        `The {person} is wearing a {color} {pattern} {sing_clothing}.`  
        `The {person} has on a {color} {pattern} {sing_clothing}.`  
        `The {person} is wearing {color} {pattern} {plur_clothing}.`  
        `The {person} has on {color} {pattern} {plur_clothing}.`  
    b. Negative  
        `The {person} is not wearing a {color} {pattern} {sing_clothing}.`  
        `The {person} does not have on a {color} {pattern} {sing_clothing}.`  
        `The {person} is not wearing {color} {pattern} {plur_clothing}.`  
        `The {person} does not have on {color} {pattern} {plur_clothing}.`  

E:
    a1 -> b1 (change)
    a2 -> b2 (change)
    a3 -> b3 (change)
    a4 -> b4 (change)
N:
    b1 -> a1 (change)
    b2 -> a2 (change)
    b3 -> a3 (change)
    b4 -> a4 (change)
C:
    a1 -> b1 (don't change)
    a2 -> b2 (don't change)
    a3 -> b3 (don't change)
    a4 -> b4 (don't change)
    b1 -> a1 (don't change)
    b2 -> a2 (don't change)
    b3 -> a3 (don't change)
    b4 -> a4 (don't change)

6. Common color _and_ uncommon pattern:

    a. Positive  
        `The {person} is wearing a {color} {uncommon_pattern} {sing_clothing}.`  
        `The {person} has on a {color} {uncommon_pattern} {sing_clothing}.`  
        `The {person} is wearing {color} {uncommon_pattern} {plur_clothing}.`  
        `The {person} has on {color} {uncommon_pattern} {plur_clothing}.`  
    b. Negative  
        `The {person} is not wearing a {color} {uncommon_pattern} {sing_clothing}.`  
        `The {person} does not have on a {color} {uncommon_pattern} {sing_clothing}.`  
        `The {person} is not wearing {color} {uncommon_pattern} {plur_clothing}.`  
        `The {person} does not have on {color} {uncommon_pattern} {plur_clothing}.`  

7. Uncommon color _and_ common pattern:

    a. Positive  
        `The {person} is wearing a {uncommon_color} {pattern} {sing_clothing}.`  
        `The {person} has on a {uncommon_color} {pattern} {sing_clothing}.`  
        `The {person} is wearing {uncommon_color} {pattern} {plur_clothing}.`  
        `The {person} has on {uncommon_color} {pattern} {plur_clothing}.`  
    b. Negative  
        `The {person} is not wearing a {uncommon_color} {pattern} {sing_clothing}.`  
        `The {person} does not have on a {uncommon_color} {pattern} {sing_clothing}.`  
        `The {person} is not wearing {uncommon_color} {pattern} {plur_clothing}.`  
        `The {person} does not have on {uncommon_color} {pattern} {plur_clothing}.`  

8. Uncommon color _and_ uncommon pattern:

    a. Positive  
        `The {person} is wearing a {uncommon_color} {pattern} {sing_clothing}.`  
        `The {person} has on a {uncommon_color} {pattern} {sing_clothing}.`  
        `The {person} is wearing {uncommon_color} {pattern} {plur_clothing}.`  
        `The {person} has on {uncommon_color} {pattern} {plur_clothing}.`  
    b. Negative  
        `The {person} is not wearing a {uncommon_color} {uncommon_pattern} {sing_clothing}.`  
        `The {person} does not have on a {uncommon_color} {uncommon_pattern} {sing_clothing}.`  
        `The {person} is not wearing {uncommon_color} {uncommon_pattern} {plur_clothing}.`  
        `The {person} does not have on {uncommon_color} {uncommon_pattern} {plur_clothing}.`  

9. Two common colors:

    a. Positive  
        `The {person} is wearing a {color} and {color} {sing_clothing}.`  
        `The {person} has on a {color} and {color} {sing_clothing}.`  
        `The {person} is wearing {color} and {color} {plur_clothing}.`  
        `The {person} has on {color} and {color} {plur_clothing}.`  
    b. Negative  
        `The {person} is not wearing a {color} and {color} {sing_clothing}.`  
        `The {person} does not have on a {color} and {color} {sing_clothing}.`  
        `The {person} is not wearing {color} and {color} {plur_clothing}.`  
        `The {person} does not have on {color} and {color} {plur_clothing}.`  

10. Common color and uncommon color:

    a. Positive  
        `The {person} is wearing a {color} and {uncommon_color} {sing_clothing}.`  
        `The {person} has on a {color} and {uncommon_color} {sing_clothing}.`  
        `The {person} is wearing {color} and {uncommon_color} {plur_clothing}.`  
        `The {person} has on {color} and {uncommon_color} {plur_clothing}.`  
    b. Negative  
        `The {person} is not wearing a {color} and {uncommon_color} {sing_clothing}.`  
        `The {person} does not have on a {color} and {uncommon_color} {sing_clothing}.`  
        `The {person} is not wearing {color} and {uncommon_color} {plur_clothing}.`  
        `The {person} does not have on {color} and {uncommon_color} {plur_clothing}.`  

11. Uncommon color and common color:

    a. Positive  
        `The {person} is wearing a {uncommon_color} and {color} {sing_clothing}.`  
        `The {person} has on a {uncommon_color} and {color} {sing_clothing}.`  
        `The {person} is wearing {uncommon_color} and {color} {plur_clothing}.`  
        `The {person} has on {uncommon_color} and {color} {plur_clothing}.`  
    b. Negative  
        `The {person} is not wearing a {uncommon_color} and {color} {sing_clothing}.`  
        `The {person} does not have on a {uncommon_color} and {color} {sing_clothing}.`  
        `The {person} is not wearing {uncommon_color} and {color} {plur_clothing}.`  
        `The {person} does not have on {uncommon_color} and {color} {plur_clothing}.`  

12. Two common colors _and_ common pattern:

    a. Positive  
        `The {person} is wearing a {color} and {color} {pattern} {sing_clothing}.`  
        `The {person} has on a {color} and {color} {pattern} {sing_clothing}.`  
        `The {person} is wearing {color} and {color} {pattern} {plur_clothing}.`  
        `The {person} has on {color} and {color} {pattern} {plur_clothing}.`  
    b. Negative  
        `The {person} is not wearing a {color} and {color} {pattern} {sing_clothing}.`  
        `The {person} does not have on a {color} and {color} {pattern} {sing_clothing}.`  
        `The {person} is not wearing {color} and {color} {pattern} {plur_clothing}.`  
        `The {person} does not have on {color} and {color} {pattern} {plur_clothing}.`  

13. Common color, uncommon color, _and_ common pattern:

    a. Positive  
        `The {person} is wearing a {color} and {uncommon_color} {pattern} {sing_clothing}.`  
        `The {person} has on a {color} and {uncommon_color} {pattern} {sing_clothing}.`  
        `The {person} is wearing {color} and {uncommon_color} {pattern} {plur_clothing}.`  
        `The {person} has on {color} and {uncommon_color} {pattern} {plur_clothing}.`  
    b. Negative  
        `The {person} is not wearing a {color} and {uncommon_color} {pattern} {sing_clothing}.`  
        `The {person} does not have on a {color} and {uncommon_color} {pattern} {sing_clothing}.`  
        `The {person} is not wearing {color} and {uncommon_color} {pattern} {plur_clothing}.`  
        `The {person} does not have on {color} and {uncommon_color} {pattern} {plur_clothing}.`  

14. Uncommon color, common color, _and_ common pattern:

    a. Positive  
        `The {person} is wearing a {uncommon_color} and {color} {pattern} {sing_clothing}.`  
        `The {person} has on a {uncommon_color} and {color} {pattern} {sing_clothing}.`  
        `The {person} is wearing {uncommon_color} and {color} {pattern} {plur_clothing}.`  
        `The {person} has on {uncommon_color} and {color} {pattern} {plur_clothing}.`  
    b. Negative  
        `The {person} is not wearing a {uncommon_color} and {color} {pattern} {sing_clothing}.`  
        `The {person} does not have on a {uncommon_color} and {color} {pattern} {sing_clothing}.`  
        `The {person} is not wearing {uncommon_color} and {color} {pattern} {plur_clothing}.`  
        `The {person} does not have on {uncommon_color} and {color} {pattern} {plur_clothing}.`  

15. Two uncommon colors _and_ common pattern:

    a. Positive  
        `The {person} is wearing a {uncommon_color} and {uncommon_color} {pattern} {sing_clothing}.`  
        `The {person} has on a {uncommon_color} and {uncommon_color} {pattern} {sing_clothing}.`  
        `The {person} is wearing {uncommon_color} and {uncommon_color} {pattern} {plur_clothing}.`  
        `The {person} has on {uncommon_color} and {uncommon_color} {pattern} {plur_clothing}.`  
    b. Negative  
        `The {person} is not wearing a {uncommon_color} and {uncommon_color} {pattern} {sing_clothing}.`  
        `The {person} does not have on a {uncommon_color} and {uncommon_color} {pattern} {sing_clothing}.`  
        `The {person} is not wearing {uncommon_color} and {uncommon_color} {pattern} {plur_clothing}.`  
        `The {person} does not have on {uncommon_color} and {uncommon_color} {pattern} {plur_clothing}.`  

16. Two common colors _and_ uncommon pattern:

    a. Positive  
        `The {person} is wearing a {color} and {color} {uncommon_pattern} {sing_clothing}.`  
        `The {person} has on a {color} and {color} {uncommon_pattern} {sing_clothing}.`  
        `The {person} is wearing {color} and {color} {uncommon_pattern} {plur_clothing}.`  
        `The {person} has on {color} and {color} {uncommon_pattern} {plur_clothing}.`  
    b. Negative  
        `The {person} is not wearing a {color} and {color} {uncommon_pattern} {sing_clothing}.`  
        `The {person} does not have on a {color} and {color} {uncommon_pattern} {sing_clothing}.`  
        `The {person} is not wearing {color} and {color} {uncommon_pattern} {plur_clothing}.`  
        `The {person} does not have on {color} and {color} {uncommon_pattern} {plur_clothing}.`  

17. Common color, uncommon color, _and_ uncommon pattern:

    a. Positive  
        `The {person} is wearing a {color} and {uncommon_color} {uncommon_pattern} {sing_clothing}.`  
        `The {person} has on a {color} and {uncommon_color} {uncommon_pattern} {sing_clothing}.`  
        `The {person} is wearing {color} and {uncommon_color} {uncommon_pattern} {plur_clothing}.`  
        `The {person} has on {color} and {uncommon_color} {uncommon_pattern} {plur_clothing}.`  
    b. Negative  
        `The {person} is not wearing a {color} and {uncommon_color} {uncommon_pattern} {sing_clothing}.`  
        `The {person} does not have on a {color} and {uncommon_color} {uncommon_pattern} {sing_clothing}.`  
        `The {person} is not wearing {color} and {uncommon_color} {uncommon_pattern} {plur_clothing}.`  
        `The {person} does not have on {color} and {uncommon_color} {uncommon_pattern} {plur_clothing}.`  

18. Uncommon color, common color, _and_ uncommon pattern:

    a. Positive  
        `The {person} is wearing a {uncommon_color} and {color} {uncommon_pattern} {sing_clothing}.`  
        `The {person} has on a {uncommon_color} and {color} {uncommon_pattern} {sing_clothing}.`  
        `The {person} is wearing {uncommon_color} and {color} {uncommon_pattern} {plur_clothing}.`  
        `The {person} has on {uncommon_color} and {color} {uncommon_pattern} {plur_clothing}.`  
    b. Negative  
        `The {person} is not wearing a {uncommon_color} and {color} {uncommon_pattern} {sing_clothing}.`  
        `The {person} does not have on a {uncommon_color} and {color} {uncommon_pattern} {sing_clothing}.`  
        `The {person} is not wearing {uncommon_color} and {color} {uncommon_pattern} {plur_clothing}.`  
        `The {person} does not have on {uncommon_color} and {color} {uncommon_pattern} {plur_clothing}.`  

19. Two uncommon colors _and_ uncommon pattern:

    a. Positive  
        `The {person} is wearing a {uncommon_color} and {uncommon_color} {uncommon_pattern} {sing_clothing}.`  
        `The {person} has on a {uncommon_color} and {uncommon_color} {uncommon_pattern} {sing_clothing}.`  
        `The {person} is wearing {uncommon_color} and {uncommon_color} {uncommon_pattern} {plur_clothing}.`  
        `The {person} has on {uncommon_color} and {uncommon_color} {uncommon_pattern} {plur_clothing}.`  
    b. Negative  
        `The {person} is not wearing a {uncommon_color} and {uncommon_color} {uncommon_pattern} {sing_clothing}.`  
        `The {person} does not have on a {uncommon_color} and {uncommon_color} {uncommon_pattern} {sing_clothing}.`  
        `The {person} is not wearing {uncommon_color} and {uncommon_color} {uncommon_pattern} {plur_clothing}.`  
        `The {person} does not have on {uncommon_color} and {uncommon_color} {uncommon_pattern} {plur_clothing}.`  
