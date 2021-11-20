# Goal

- Benchmark OpenAI's GPT3 models with NLI and/or QA dataset to establish a baseline.
- Evaluate the models weaknesses using one of the challenge sets and/or adversarial attacks.
- Analyse where the model could see big improvements.
- Experiment with different prompt engineering to augment the dataset.
- Identify dataset artifacts that renders certain prompts very ineffective.


# Implementation

## Establishing the Baseline

Use OpenAI's example prompt for Q&A for establishing a benchmark for each models.

**Prompt**
```
I am a highly intelligent question answering bot. If you ask me a question that is rooted in truth, I will give you the answer. If you ask me a question that is nonsense, trickery, or has no clear answer, I will respond with "Unknown".

Q: What is human life expectancy in the United States?
A: Human life expectancy in the United States is 78 years.

Q: Who was president of the United States in 1955?
A: Dwight D. Eisenhower was president of the United States in 1955.

Q: Which party did he belong to?
A: He belonged to the Republican Party.

Q: What is the square root of banana?
A: Unknown

Q: How does a telescope work?
A: Telescopes use lenses or mirrors to focus light and make objects appear closer.

Q: Where were the 1992 Olympics held?
A: The 1992 Olympics were held in Barcelona, Spain.

Q: How many squigs are in a bonk?
A: Unknown

Q: Where is the Valley of Kings?
A:
```

**Sample response**
```
The Valley of Kings is in Luxor, Egypt.
```

This requires us to:
1. Prepare the Q&A dataset with the example prompt
2. Build the API harness for making inference on GPT3 models
3. Parse the response output and check for reliability
4. Evaluate the performance and compare it against other models

OpenAI provides an easy to use API interface to work with. We'll start with the fastest/cheapest model available for development and then consider using more expensive models for ablation experiments.

In the example prompt, we need to provide several Q&A examples, so that the model can effective learn from the examples on the spot. There are ways to fine tune the GPT3 model, but we'll start with few-shot approach first since that's the main strength of GPT3 models.

It's not clear whether the examples has to be fixed set or a randomly sampled set. The most important requirement is that the test set is not part of the examples.

It doesn't looks like there's a way to batch the input, so it might take quite some time to evaluate the dataset. 

There are parameters such as `Max tokens`, `temperature`, `Top p`, `Frequency`, etc. We aren't sure what the best values are for these hyperparameters just yet.

## Getting Started with API

We use the example prompt provided by OpenAI for Q&A to evalute the GPT3 models against the suggested datasets (SQuAD and HotpotQA).

The naive approach of simply copying and pasting the question doesn't work well. The models does answer the question correctly, but the response is ususally in complete sentence whereas the dataset expects a simplest answer.

Also, the example prompt doesn't have a way of providing a context, so this needs to be modified with some data points from the datasets to introduce context to the prompt.
- This requires using model tokens, so eventually, it might be beneficial to fine tune the model, so that we don't have to provide examples.

Without fine-tuning or prompt engineering, the model refuse to generate any text.

## Q&A vs NLI

Q&A presents a challenge due to the greater degree of freedom in terms of what's expected as an output. It makes it harder to evaluate because the response could be very close to right but different enough that it's not easy to say whether the model got it right or not.

NLI on the other hand is much simpler. The model can only respond with neutral, entailment, and contradtiction. This makes it much easier to evaluate and establish certain baseline and develop a proof of concept.

