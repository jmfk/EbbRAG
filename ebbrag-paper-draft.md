# EbbRAG: Spaced Repetition and Active Recall as Retrieval Mechanisms in RAG Systems

**Johan [Efternamn]**
Independent Researcher

*Preprint — arXiv submission draft*
*June 2026*

## Abstract

Retrieval-Augmented Generation (RAG) systems typically treat retrieval as a stateless, similarity-only operation: every query is answered by re-ranking a fixed corpus purely on semantic distance, with no notion of which facts the system has reinforced through repeated use, and no attempt by the model to consult its own parametric memory before reaching for external context. We introduce **EbbRAG**, a RAG architecture that incorporates two cognitively-inspired mechanisms from human memory research: (1) **Spaced Repetition Scoring (SR-Score)**, which models each chunk's retrieval-worthiness as a function of an Ebbinghaus-style forgetting curve, reinforced by repeated retrieval and decayed by disuse, and (2) **Pre-Retrieval Active Recall (AR-Step)**, in which the generator first produces a candidate answer from parametric memory and uses it to bias retrieval toward chunks consistent with — or correcting — that recollection. We evaluate EbbRAG against VanillaRAG, BM25RAG, and a prompt-engineered SelfRAG baseline on Natural Questions, HotpotQA, and a new synthetic benchmark, EbbRAG-Temporal, designed to probe retrieval quality as a corpus ages without re-indexing. We hypothesize that SR-Score reduces temporal degradation of retrieval precision and that the AR-Step reduces hallucination rate by anchoring retrieval to the model's existing beliefs. We report our experimental design, hyperparameters, and ablation methodology in full so that results can be reproduced and extended.

## 1. Introduction

Standard RAG (Lewis et al., 2020) pairs a frozen or fine-tuned retriever with a generator, treating every query independently of retrieval history. This has two consequences that we argue are sub-optimal: first, frequently-useful chunks receive no special treatment over rarely-useful ones beyond their static embedding similarity to the current query, even though usage patterns are a strong, freely-available signal of relevance; second, the generator's own parametric knowledge is discarded prior to retrieval rather than used to inform it, even in cases where the model already "knows" the answer and retrieval serves only to verify or refine it.

Human long-term memory research offers two well-studied phenomena that map naturally onto these gaps. The **spacing effect** (Ebbinghaus, 1885; Wozniak & Gorzelanczyk, 1994) shows that memories which are recalled at expanding intervals are retained far longer than those recalled at fixed or random intervals, and that recall itself reinforces retention — a property exploited by spaced-repetition schedulers such as the Leitner system (Leitner, 1972). **Active recall** / **retrieval practice** (Roediger & Karpicke, 2006) shows that the act of attempting to recall information from memory before being shown the answer improves subsequent retention and understanding more than passive re-exposure does.

We translate both mechanisms into the retrieval step of a RAG pipeline. EbbRAG maintains a per-chunk *stability* parameter that increases with use and is fed into a retention-style decay function (SR-Score) which is blended with cosine similarity at query time. Separately, before retrieving, EbbRAG asks the generator to produce a brief candidate answer from memory alone; the embedding of that candidate is blended into the retrieval score (AR-Step), nudging retrieval toward chunks that either corroborate or contradict the model's prior belief.

Our contributions:
1. SR-Score, a lightweight, model-agnostic chunk-scoring mechanism inspired by the Ebbinghaus forgetting curve, requiring no additional training.
2. The AR-Step, a pre-retrieval active-recall mechanism that conditions retrieval on the generator's own parametric answer.
3. EbbRAG-Temporal, a synthetic benchmark built by replicating NQ queries across four time deltas (0, 7, 30, 90 days) without re-indexing the corpus, designed to isolate temporal degradation effects.
4. A full ablation methodology (decay rate λ, AR mixing weight α3, retrieval depth k) and baseline suite (VanillaRAG, BM25RAG, SelfRAG, SR-only, AR-only) for isolating each component's contribution.

## 2. Related Work

### 2.1 RAG Architectures

Lewis et al. (2020) introduced RAG as a differentiable combination of a dense retriever and a seq2seq generator. Subsequent work has largely focused on improving the retriever (better embeddings, hybrid lexical/dense retrieval) or the generator's use of retrieved context, but has generally preserved the assumption that retrieval is a stateless function of the query alone.

### 2.2 Memory-Augmented LLMs

Systems such as MemGPT (Packer et al., 2023) introduce explicit memory hierarchies (working memory vs. archival storage) and let the model manage what is paged in and out, but do not model the *temporal dynamics* of which memories should be considered "fresh" versus "stale" — memory promotion/demotion is driven by capacity pressure, not by a forgetting-curve-style decay function tied to usage.

### 2.3 Self-Knowledge and Pre-Retrieval Generation

Self-RAG (Asai et al., 2024) trains a model to emit reflection tokens that decide whether to retrieve, critique retrieved passages, and assess whether the final answer is supported — closely related in spirit to our AR-Step, but reactive (deciding whether/how to use retrieval) rather than using a self-generated candidate to actively reshape the retrieval ranking itself. FLARE (Jiang et al., 2023) triggers retrieval based on the generator's own token-level confidence during generation, which is a related but orthogonal "actively query when uncertain" mechanism rather than a recall-then-bias-retrieval mechanism applied up front.

### 2.4 Cognitive Foundations

Ebbinghaus (1885) established the forgetting curve and the benefit of spaced review; Leitner (1972) operationalized spaced repetition into a practical scheduling system; Wozniak & Gorzelanczyk (1994) formalized the SuperMemo family of spaced-repetition algorithms used in modern flashcard software; Roediger & Karpicke (2006) demonstrated the testing effect, showing that active recall outperforms passive restudy for long-term retention. EbbRAG is, to our knowledge, the first system to translate both the spacing effect and the testing effect directly into the retrieval scoring function of a RAG pipeline.

## 3. Method

### 3.1 Background: Standard RAG

Given query $q$, a standard dense retriever scores each chunk $c_i$ in corpus $C$ by cosine similarity between query and chunk embeddings:

$$ s_i = \frac{e_q \cdot e_{c_i}}{\lVert e_q \rVert \lVert e_{c_i} \rVert} $$

The top-$k$ chunks by $s_i$ are passed to the generator along with $q$ to produce the final answer.

### 3.2 Component 1 — Spaced Repetition Scoring (SR-Score)

Each chunk $c_i$ carries a stability parameter $\sigma_i$ (initialized to $\sigma_0 = 0.5$) and a timestamp of last retrieval $\tau_i$. At query time $t$, we define a retention-style weight:

$$ R_i(t) = \epsilon + (1 - \epsilon) \cdot \exp\left(-\lambda \cdot \frac{t - \tau_i}{\max(\sigma_i, \epsilon)}\right) $$

where $\lambda$ is a global decay-rate hyperparameter and $\epsilon$ is a floor preventing the weight from collapsing to zero for very stale chunks. The final retrieval score blends similarity and retention:

$$ \text{score}_i = \alpha_1 \cdot s_i + \alpha_2 \cdot R_i(t) $$

After a chunk is retrieved, its stability is reinforced proportionally to how long it had been since its last retrieval (mirroring the spacing effect — recall after a longer gap produces a larger stability gain):

$$ \sigma_i \leftarrow \sigma_i + \log\left(1 + \frac{t - \tau_i}{\Delta_{\text{day}}}\right), \qquad \tau_i \leftarrow t $$

### 3.3 Component 2 — Pre-Retrieval Active Recall (AR-Step)

Before retrieval, the generator is prompted to produce a brief candidate answer $\hat{a}$ from parametric memory alone, with no context supplied. We embed $\hat{a}$ and compute its similarity to each chunk, adding a third term to the retrieval score:

$$ \text{score}_i = \alpha_1 \cdot s_i + \alpha_2 \cdot R_i(t) + \alpha_3 \cdot \frac{e_{\hat{a}} \cdot e_{c_i}}{\lVert e_{\hat{a}} \rVert \lVert e_{c_i} \rVert} $$

Intuitively, this nudges retrieval toward passages that either corroborate the model's existing belief (when correct) or that are most likely to surface a correction (when the chunk corpus contains the actual answer and the candidate is wrong but topically adjacent).

### 3.4 Combined System: EbbRAG

EbbRAG combines both components with default weights $\alpha_1 = 0.6$, $\alpha_2 = 0.2$, $\alpha_3 = 0.2$, selected via the ablation sweeps in Section 6 / SPEC-05. The full pipeline:

1. Receive query $q$ at time $t$.
2. Generate candidate answer $\hat{a}$ from parametric memory (AR-Step).
3. Embed $q$ and $\hat{a}$.
4. Compute $\text{score}_i$ for every chunk using similarity, SR-Score, and AR similarity.
5. Select top-$k$ chunks by $\text{score}_i$.
6. Update stability $\sigma_i$ and last-retrieved timestamp $\tau_i$ for each selected chunk.
7. Generate the final answer conditioned on $q$ and the selected chunks.
8. Return the final answer, the candidate answer, and retrieval metadata for evaluation.

## 4. Experimental Setup

### 4.1 Datasets

| Dataset | Role | Size (eval) | Notes |
|---|---|---|---|
| Natural Questions (NQ-Open) | Primary QA benchmark | NQ-dev, linked subset | Queries linked to Wikipedia chunks via answer-substring matching |
| HotpotQA (distractor) | Multi-hop QA benchmark | distractor validation split | Tests retrieval under multi-hop, distractor-heavy conditions |
| EbbRAG-Temporal | Temporal degradation probe | linked NQ queries × 4 time deltas | Same query replicated at t = 0, 7, 30, 90 days with no re-indexing between timesteps |

### 4.2 Baselines

| System | Retrieval | SR-Score | AR-Step | Notes |
|---|---|---|---|---|
| VanillaRAG | Dense cosine, top-k | No | No | Primary baseline; equivalent to EbbRAG(use_sr=False, use_ar=False) |
| BM25RAG | Lexical (BM25Okapi) | No | No | Tests whether SR gains are retrieval-method agnostic |
| SelfRAG (prompt-engineered) | Dense cosine + relevance filter | No | No | Simplified Self-RAG without fine-tuning; weaker than Asai et al. (2024) |
| SR-only | Dense cosine + SR-Score | Yes | No | Isolates the spacing-effect contribution |
| AR-only | Dense cosine + AR-Step | No | Yes | Isolates the active-recall contribution |
| EbbRAG (full) | Dense cosine + SR-Score + AR-Step | Yes | Yes | Combined system |

### 4.3 Models

- **Retriever:** all-MiniLM-L6-v2 (and BAAI/bge-small-en-v1.5 in implementation notebooks)
- **Generator:** claude-sonnet-4-6

### 4.4 Metrics

| Metric | Definition |
|---|---|
| Recall@k | Fraction of queries for which at least one gold-relevant chunk appears in the top-k retrieved set |
| Exact Match (EM) | 1 if the normalized gold answer is a substring of the normalized model answer, else 0 |
| F1 | Token-level F1 between predicted and gold answer |
| Hallucination Rate | Fraction of answers asserting facts not supported by any retrieved chunk |
| Retrieval Precision@k | Fraction of retrieved chunks that are gold-relevant |
| Temporal Degradation Slope | Slope of EM vs. time-delta on EbbRAG-Temporal; measures how quickly retrieval quality decays as the corpus ages without reinforcement |

### 4.5 Hyperparameters

Default: $\lambda = 0.05$, $\epsilon = 0.01$, $\alpha_1 = 0.6$, $\alpha_2 = 0.2$, $\alpha_3 = 0.2$, $k = 5$. Ablation sweeps: $\lambda \in \{0.01, 0.05, 0.1, 0.5\}$; $\alpha_3 \in \{0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0\}$; $k \in \{1, 3, 5, 10, 20\}$.

## 5. Hypotheses

- **H1:** EbbRAG achieves higher EM/F1 than VanillaRAG and BM25RAG on NQ and HotpotQA.
- **H2:** EbbRAG exhibits a flatter Temporal Degradation Slope than VanillaRAG on EbbRAG-Temporal, i.e., SR-Score mitigates retrieval staleness without requiring re-indexing.
- **H3:** AR-only and EbbRAG (full) exhibit lower Hallucination Rate than VanillaRAG, because the AR-Step anchors retrieval to the model's own prior belief.
- **H4:** The benefit of SR-Score is retrieval-method agnostic, i.e., comparable relative gains appear whether the underlying retriever is dense (cosine) or lexical (BM25).

## 6. Expected Results and Analysis

*[Results table to be populated after full experimental runs — placeholder]*

| System | NQ EM | NQ F1 | HotpotQA EM | Hallucination Rate | Temporal Slope |
|---|---|---|---|---|---|
| VanillaRAG | — | — | — | — | — |
| BM25RAG | — | — | — | — | — |
| SelfRAG | — | — | — | — | — |
| SR-only | — | — | — | — | — |
| AR-only | — | — | — | — | — |
| EbbRAG (full) | — | — | — | — | — |

We expect the ablation sweeps (Section 4.5) to show that EM is relatively insensitive to $\lambda$ within a moderate range but degrades sharply for very large $\lambda$ (over-aggressive decay collapses SR-Score to noise) and very small $\lambda$ (SR-Score becomes nearly static, losing its differentiation power). We expect a similar inverted-U pattern for $\alpha_3$: too small and the AR-Step has no effect; too large and retrieval over-anchors on a possibly-wrong candidate answer, hurting precision.

## 7. Limitations

- The SelfRAG baseline is a prompt-engineered approximation, not the fine-tuned model from Asai et al. (2024), and likely understates true Self-RAG performance.
- The substring-matching heuristic used to link NQ queries to "relevant" Wikipedia chunks is a weak label and may introduce noise into Recall@k and Precision@k.
- EbbRAG-Temporal is synthetic (replicated queries at fixed deltas rather than a naturally evolving corpus), and may not capture the full complexity of real-world corpus drift.
- The AR-Step adds an extra generation call per query, increasing latency and cost relative to VanillaRAG; this tradeoff is not yet quantified.

## 8. Discussion

EbbRAG's two components are designed to be independently switchable and additively composable, which is reflected in the baseline suite (SR-only, AR-only, full). This lets us attribute gains (or regressions) to each mechanism individually rather than only evaluating the combined system. The use of cognitively-motivated update rules (forgetting-curve decay, spacing-effect reinforcement, testing-effect-style pre-retrieval recall) is intended as a principled, low-overhead alternative to learned re-ranking or fine-tuned retrieval-control policies, trading some of their flexibility for interpretability and zero additional training cost.

## 9. Conclusion

We presented EbbRAG, a RAG architecture that incorporates spaced-repetition-style retrieval scoring and a pre-retrieval active-recall step, both inspired directly by established findings in human memory research. We described a full experimental protocol — including a new temporal-degradation benchmark and a baseline/ablation suite — for evaluating whether these cognitively-inspired mechanisms improve answer quality, reduce hallucination, and slow the staleness of retrieval as a corpus ages without re-indexing. We leave full empirical results to a forthcoming revision of this draft.

## References

- Asai, A., et al. (2024). Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection.
- Ebbinghaus, H. (1885). Über das Gedächtnis: Untersuchungen zur experimentellen Psychologie.
- Jiang, Z., et al. (2023). Active Retrieval Augmented Generation (FLARE).
- Kwiatkowski, T., et al. (2019). Natural Questions: A Benchmark for Question Answering Research.
- Leitner, S. (1972). So lernt man lernen.
- Lewis, P., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.
- Packer, C., et al. (2023). MemGPT: Towards LLMs as Operating Systems.
- Roediger, H. L., & Karpicke, J. D. (2006). Test-Enhanced Learning: Taking Memory Tests Improves Long-Term Retention.
- Yang, Z., et al. (2018). HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering.
- Wozniak, P. A., & Gorzelanczyk, E. J. (1994). Optimization of Repetition Spacing in the Practice of Learning.

---

Word count (excl. references): ~3,800
Target: arXiv cs.IR / cs.CL