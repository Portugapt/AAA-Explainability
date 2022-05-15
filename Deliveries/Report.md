# A Study of SHAP and LIME, comparisons and limitations in two datasets.

- [A Study of SHAP and LIME, comparisons and limitations in two datasets.](#a-study-of-shap-and-lime-comparisons-and-limitations-in-two-datasets)
  - [Abstract](#abstract)
  - [What is LIME?](#what-is-lime)
  - [What is SHAP?](#what-is-shap)
    - [Questions](#questions)
    - [Introduction](#introduction)
    - [Additive Feature Attribution Methods](#additive-feature-attribution-methods)
      - [Definition 1: **Additive feature attribution methods**](#definition-1-additive-feature-attribution-methods)
      - [LIME](#lime)
      - [DeepLIFT](#deeplift)
      - [Layer-Wise Relevance Propagation](#layer-wise-relevance-propagation)
      - [Classic Shapley Value Estimation](#classic-shapley-value-estimation)
        - [Shapley regression values](#shapley-regression-values)
  - [Introduction](#introduction-1)
  - [Problem definition](#problem-definition)
  - [Challenges](#challenges)
  - [Results](#results)
  - [References](#references)

## Abstract

## What is LIME?

## What is SHAP?

SHAP assigns each feature an importance value for a particular prediction. 

Novel components

- Identification of a new class of additive feature importance measures  
- theoretical results showing there is a unique solution in this class with a set of desirable properties.

This set of desirable properties is missing in recent proposals. 
### Questions

What is the set of desirable properties?

How does this mapping function works?

What is going to be $g(z')$ in definition 1?

* Cooperative game theory equations
  * classic equations
  * Shapley regression values
  * Shapley sampling values
  * quantitative input influence

### Introduction

Interpretation of prediction model's output garners user trust, provides insight into how the model may be improved, and helps understanding of the process being modeled. Sometimes, model engineering gravitates towards simpler models with more interpretability in a trade-off for more complex models with more accuracy.  

Findings:  

1. Any explanation of a model's prediction as a model itself. (*explanation model*) Six current methods unified which the authors defined as *additive feature attribution methods* (Section 2)
2. Garantee of a unique solution, coming from game theory, in the class of additive feature attribution methods (Section 3), and SHAP *values* as the unified measure of importance for those 6 methods approximations. 
3. SHAP value estimation methods are better aligned with human intuition, and effectually discriminate model output classes. (Section 5)

### Additive Feature Attribution Methods

We must create a simpler *explanation model* for more complex models, which is defined as an interpretable approximation of the original model. Six current explanation methods from literature use the same *explanation model*, and this unity of all them has not been appreciated.

------

Local methods

- Let $f$ be the original prediction model.
- Let $g$ be the explanation model.
- LIME: Explanation of prediction $f(x)$ based on single input $x$

Often, explanation models:  

- *Simplified inputs* $x{'}$, map original input through mapping function $x = h_{x}(x')$
- Local methods, $g(z') \approx f(h_{x}(z'))$, whenever $z' \approx x'$
- Note: $h_{x}$ is specific to current input $x$
- Note: So $h_{x}(x') = x$, even though $x'$ contains less info than $x$

#### Definition 1: **Additive feature attribution methods**  

Linear function of **binary variables** 

* $z' \in \{0, 1\}^{M}$ (either 0 or 1)
* $M$ is the the number of simplified input features
* $\phi_{i} \in \mathbb{R}$ (any number)

$$ g(z') = \phi + \sum_{i=1}^{M} \phi_{i}z'_{i} $$
(Equation 1)

So, it's $\phi_{0}$ + the weight of each $\phi$, if the feature $z'_{i}$ is present in the simplified features. (1 -> Present, 0 -> Not Present). $g(z')$ is the *explanation model*

#### LIME

The LIME method interprets individual model predictions based on locally approximating the model around a given prediction.

* simplified inputs $x'$ as "interpretable inputs"
* Mapping $x = h_{x}(x')$ converts binary vector of "interpretable inputs" into original input space.
* Different types of $x = h_{x}(x')$ for different input spaces.

Objective function:
$$\xi = \underset{g \in \varrho}{arg min} L(f, g, \pi_{x'} + \Omega(g)$$  
(Equation 2)

* Faithfulness (Approximation to) of the **explanation model** $g(z')$ to the **original model** $f(h_{x}(z'))$ is enforced through the loss $L$ (Squared Loss) over a set of samples in the *simplified input space* $(x')$
* Weighted by the local kernel $\pi_{x'}$.
* $\Omega$ penalizes the complexity of g.
* Equation 2 can be solved using penalized linear regression.

#### DeepLIFT

Recursive prediction explanation method for deep learning.  
Input $x_i$ has a value $C_{\Delta x_i \Delta y}$ that is equal to effect of that input being set to a **reference value**. (Default value) 
The mapping $x = h_{x}(x')$ converts binary values into the original inputs. 1 -> Input takes original value, 0 -> Input takes reference value.
> The reference value, though chosen by the user, represents a typical uninformative background value for the feature.

"summation-to-delta" property that states:  
$$\sum_{i=1}^{n} C_{\Delta x_i \Delta o} = \Delta_O$$
(Equation 3)

* $o = f(x)$ --> Model Output
* $\Delta_O = f(x) - f(r)$
* $\Delta x_i = x_i - r_i$ 
* r --> Referenee input

If then: 

* $\phi_i = C_{\Delta x_i \Delta o}$ 
* $\phi_0 = f(r)$

We get equation (1) again, making DeepLIFT another additive feature attribution method.

#### Layer-Wise Relevance Propagation

> This menthod is equivalent to DeepLIFT with the reference activations of all neurons fixed to zero. 

We can see it as a special case of DeepLIFT. The equation also matches equation (1)

#### Classic Shapley Value Estimation

There are three methods that use classic equations from cooperative game thoory to compute explanatations of model predictions:
* Shpaley regression values
* Shapley sampling values
* Quantitative Input Influence

##### Shapley regression values

* $F$ is the set of all fetures.
* Retraining of the model on all feature subsets $S \subseteq F$
* Model $f_{S \cup \{i\}}$ trained with feature present; Model $f_S$ with feature withheld;
* Compare models on current input $f_{S \cup \{i\}}(x_{S \cup \{i\}}) - f_{S} (x_S)$; $x_S$ are the values of the input features in the set $S$
* Effect of withholding a feature is calculated.
* The preciding differences are computed for all possible subsets $S \subseteq F / \{i\}$

They are a weighted average of all possible differences:
![imgshapreg](https://i.imgur.com/JGRKbRy.png)

## Introduction

This project consists in making a study of the available and most used model explanation tools that currently exists. Namely, SHAP <a href="#Shap-ref-1">\[1\]</a>

c
## Problem definition

Another question we tried to answer was if adding or removing a low importance feature impacted the importance of other features in a significant way. 

> <<((Maneira interessante de adicionar estatística aqui no meio, possivelmente vai dar para comparar os resultados de um antes e depois, e se a diferença der uma distribuição é porque tem impacto, se a diferença der um gráfico aleatorio, então é porque não tem impacto. Deve ser mais ou menos isto. É possível adicionar testes de hipoteses aqui no meio, o que seria mel.)) 

Se a questão anterior for que não há diferenças significativas:
One other question we tried to answer was if we train the model with two highly correlated features, something that didn't happen in normal circumstances, will it impact the previosly present feature importance?
## Challenges

## Results

## References

<a href="https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html">Lundberg, Scott M., and Su-In Lee. "A unified approach to interpreting model predictions." Advances in neural information processing systems 30 </a>[(2017)](#Shap-ref-1)


[1]: https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html
