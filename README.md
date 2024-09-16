# mindex
a local semantic search engine of your mind index.

### motivation
i read a lot. papers, blog posts, twitter threads, you name it. a recurring problem i face is recalling information i know i've read somewhere but struggling to find it. to solve this, i built **mindex**. a local semantic search engine for your mind index (where "mind index" refers to things i (you) have read).

this project also gave me a good excuse to gain experience with embedded search. i made the most of this opportunity to learn what works and what doesn't, going to the extent of creating a synthetic benchmark dataset and performing quantitative evaluations of chunking strategies, embedding strategies, and more.

_mindex is not a rag system. instead, it provides documents and passages which best match your query. while i have little interest in attaching an llm to the end of the pipeline, everything is extendable, so go ahead and build!_

### features

- multiple search strategies:
  - okapi bm25 for lexical matching
  - dense retrieval using mixedbread-embed-large
  - hybrid search options:
    - sequential: bm25 followed by embedding search
    - reciprocal rank fusion (rrf): combines rankings from both methods
    - linear combination (lc): merges normalized scores
- colbert re-ranking using answerai-colbert-small
- support for html and pdf documents with automatic parsing
- optimized chunking strategy (600 words, 60% overlap)
- efficient vector storage and retrieval
- cli and python api for easy integration
- benchmark dataset for continuous evaluation and improvement

### usage

**cli**

typical usage would be in form of the cli. to install run

```bash
git clone https://github.com/LeonEricsson/mindex
cd mindex
pip install .
```
now you can initialize a mindex instance, add urls, search and more

```
mindex create INDEX_NAME

mindex add URL
mindex search "query"

mindex --help
mindex COMMAND --help

// commands default to using Mindex: 'default'.
```

**code**

otherwise you may use it as a class

```python
from mindex import Mindex

mindex = Mindex("my_index")

mindex.add(urls=["https://example.com/document1", "https://example.com/document2"])

results = index.search("your query here", top_k=5)

index.save("my_index.pkl")

mindex = Mindex.load("my_index.pkl")
```

### logbook
*the following sections describe the project's development process and are not necessary for using mindex.*

this entire project was born and raised in a jupyter notebook. if you're curious about the journey from a vague idea to what you see now, check out `placeholder.ipynb`. it's like a time capsule of the project's evolution.

i used the notebook as a playground - testing stuff out, creating bits and pieces, polishing things up before they graduated to proper files. it might be a bit messy at times, but it's honest work.

throughout the notebook, you'll find a ton of markdown sections. i discuss implementation alternatives, showing what worked (and what  didn't), and documenting the problems i stumbled into along the way. it's part diary, part technical doc, and maybe a little bit of me talking to myself.

so if you're into seeing how the sausage is made, or just want to laugh at my mistakes, dive into that notebook.

### benchmark
creating a robust benchmark dataset is crucial for evaluating mindex's performance. the process involved generating query-answer pairs using claude (sonnet 3.5), with the full prompt available in `data/claude_prompt.txt`.

key technical aspects of the dataset creation:

1. query generation: aimed for a mix of factual (40%), conceptual (40%), and analytical (20%) questions.
2. answer extraction: verbatim from source documents, typically 1-5 sentences long.
3. diversity constraint: only 30% of queries could directly mention the main contribution of a paper.
4. context simulation: queries framed as if posed to a large knowledge base.
5. coverage: ensured queries spanned different sections of source documents.

the main challenges involved steering claude away from generating queries that referenced specific papers or methods, and balancing query generality with specificity to create realistic scenarios. ensuring extracted answers paired with distinct passages while remaining comprehensive also proved difficult.

the final dataset comprises over 400 samples, split into validation and test sets, covering various ML topics. this benchmark dataset provides a foundation for thorough evaluation and targeted improvements of mindex's performance.

### evaluation
_i've concluded the evaluation work. ultimately we managed to improve the top-5 accuracy from 49% to 72%._

mindex is evaluated on it's [benchmark dataset](#benchmark) using two complementary metrics:

1. **top-k accuracy**: measures if any of the retrieved chunks contains the answer, using a 0.7 threshold. a score of 1 indicates successful retrieval, 0 otherwise.
2. **mean n-gram overlap score**: calculates the average of the best chunk's n-gram overlap score per retrieval, providing more nuanced results.

the complete benchmark table is available [here](/data/benchmark_results.md).

**baseline** implementation was evaluated at 46.24% accuracy with a 0.5093 mean n-gram overlap.

**chunking & cleaning.** we conducted a grid search across chunk sizes 100-800 words, and overlap percentages 20-80%. the search favors larger chunks (not a surprise), which perform better, with optimal overlap ranging from 40% to 60%. the best performer used 800-word chunks with 40% overlap, achieving 63.91% accuracy and a 0.6279 mean overlap score.

800 words is too large a chunk for a search engine result, so we perform a focused grid search in the 500-600 word range with 40-60% overlap. this revealed an optimal configuration of 600-word chunks with 60% overlap (360 words), yielding 62.41% accuracy and a 0.6216 mean n-gram overlap score—a 16-point improvement over the baseline.

attempts at implementing document cleaning within mindex were unsuccessful due to the inherent complexity of parsing PDFs. even seemingly simple tasks, such as identifying paragraphs without visual information, proved to be nearly impossible to achieve reliably. the lack of standardization in PDF structure and the focus on visual layout rather than semantic content made it challenging to develop a universal, simple cleaning strategy that would work across various document sources.

future improvements in mindex will therefore focus on more advanced document parsing techniques. PDF analyzers like [surya](https://github.com/VikParuchuri/surya) and custom TeX parsers could significantly enhance our chunking strategy by allowing for more intelligent content extraction based on document structure rather than arbitrary word counts. this is particularly valuable for PDFs, which often lack clear structural indicators, use custom encodings, and may have inconsistent formatting, leading to artifacts in extracted text.

**embeddings.** we experimented with various embedding models to improve retrieval performance. the mixedbread-ai/mxbai-embed-large-v1 model with an embedding dimension of 1024 achieved the best results, reaching 64.29% accuracy with 600-word chunks and 60% overlap. this outperformed other models like stella and gte-large. increasing the embedding dimension from 512 to 1024 provided a significant boost in performance. naturally, the expected query prefix for mxbai ("Represent this sentence for searching relevant passages:") is important for performance (and I forgot about it initially).

**hybrid search.** combining bm25 and embedding-based search yielded significant improvements. we explored three hybrid approaches:

1. sequential: use bm25 to retrieve an initial set, then apply embedding search.
2. reciprocal rank fusion (rrf): combine rankings from bm25 and embedding search.
3. linear combination (lc): combine normalized scores from both methods.

the lc approach performed best, achieving 70.68% accuracy when retrieving 100 results and selecting the top 5. this hybrid method leverages the strengths of both lexical and semantic search, resulting in more accurate retrievals. surprisingly, the sequential approach, despite its simplicity, also showed strong performance and was 212.5% faster than pure embedding search. however, the absolute time difference (0.02s to 0.0064s) is negligible for our application.

**re-rankers.** to further improve performance, we introduced a re-ranking step using the answerai-colbert-small model. this approach involves:

1. retrieving a larger set of results (top_l) using the hybrid search.
2. re-ranking these results using the colbert model.
3. selecting the final top_k results.

the best configuration achieved 72.18% accuracy with top_k=5 and top_l=20, using the hybrid lc method followed by re-ranking. interestingly, we found that increasing top_l beyond 20 didn't yield significant improvements, suggesting that the re-ranker is effective at identifying relevant results even from a relatively small candidate set.

we observed that the re-ranker became a bottleneck in performance. when increasing top_k to 20 without re-ranking, the hybrid lc method achieved 80.83% accuracy. this suggests that while the re-ranker improves precision for smaller result sets, there's potential for more advanced re-ranking models to further boost performance, especially for larger result sets.

another intriguing finding was that bm25 alone, with a top_k of 30, achieved 76.46% accuracy. traditional lexical search methods are class.

### concluding thoughts

- chunking and cleaning properly is very hard, pdfs are a mess. i honestly
think vlms like *colpali* are the way forward here.
- bm25 goes a long way in search systems, and it's very fast
- hybrid search is a nice performance boost
- a synthetic benchmark is vital. you need some metric to understand if you're improvements are worth the time and effort (obv)


