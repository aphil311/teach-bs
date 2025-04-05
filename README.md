# Teach BS

## Fluency Scoring
For fluency scoring we use a combination of "hard" structural scoring and "soft" probabilistic scoring from a distilled BERT model.

For probabilitistic scoring we begin by tokenizing the sentence and grouping words. Given the sentence `That was unbelievable` we tokenize

$$\begin{matrix} e(\text{That}) & e(\text{was}) & e(\text{un}) & e(\text{believ}) & e(\text{able}) \\ 0 & 1 & 2 & 3 & 4\end{matrix}$$

and produce groupings from the offsets

$$\mathcal G_0 = \{0\} \quad \mathcal G_1 = \{1\} \quad \mathcal G_2 = \{2, 3, 4\}.$$

We can then mask entire groups and take the average of the log probabilities of each token, negating so that lower probabilities increase loss:

$$\ell_w = - \frac 1N \sum_{i \in \mathcal G_w} \log \mathbb P(t_i)$$

where $w$ is the index of the relevant word. This penalizes words that are not preferred by the model in context, but along the way will penalize words that may be correctly used but are simply uncommon. We seek to penalize based on the "contextual awkwardness" of some given word so we will compute a rarity score

$$\mathcal F_w  = -\log (f_w + 10^{-12})$$

where $f_w$ is computed from the python `wordfreq` package. The funtion `word_frequency` will return a value $f_w \in [0, 1]$ where $0$ is extremely rare and $1$ is extremely common. Therefore $\mathcal F_w \in [0, \log 10^{-12} \approx 27.6]$ where a higher score means the word is more rare. We use this to compute an adjusted pseudo log loss $(\text{PLL})$ $\mathcal L$

$$\mathcal L_w = \ell_w - \alpha \mathcal F_w \approx \ell_w + \alpha \log f_w$$

which applies a downward adjustment to the loss for rare words where $\alpha$ is some weight parameter. Our final step is to produce an adjusted pseudolikelihood estimation

$$\mathcal J_{\text{adj}} = \frac {1} {\mathbb W} \sum_{w \in \mathbb W} \mathcal L_w.$$

We then generate a fluency score $\text{FS}$ from $0$ to $100$ using the logistic function

$$\text{FS}_{\mathbb W} = \frac{100}{1+\exp(s \times \mathcal J_{\text{adj}} - m)}$$

where $s$ is some steepness factor and $m$ is the midpoint (where the score should be $50$).