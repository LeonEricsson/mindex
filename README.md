# mindex
a local semantic search engine of your mind index.

[video]

### motivation
i read a lot. papers, blog posts, twitter threads, you name it. a recurring problem i face is recalling information i know i've read somewhere but struggling to find it. to solve this, i built **mindex**. a local semantic search engine for your mind index (where "mind index" refers to things i (you) have read).

this project also gave me a good excuse to gain experience with embedded search. i made the most of this opportunity to learn what works and what doesn't, going to the extent of creating a synthetic benchmark dataset and performing quantitative evaluations of chunking strategies, embedding strategies, and more.

_mindex is not a rag system. instead, it provides the exact passage(s) that 'best' match your query. while i have little interest in attaching an llm to the end of the pipeline, everything is extendable, so go ahead and build!_

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
this entire project was born and raised in a jupyter notebook. if you're curious about the journey from a vague idea to what you see now, check out `placeholder.ipynb`. it's like a time capsule of the project's evolution.

i used the notebook as a playground - testing stuff out, creating bits and pieces, polishing things up before they graduated to proper files. it might be a bit messy at times, but it's honest work.

throughout the notebook, you'll find a ton of markdown sections. i discuss implementation alternatives, showing what worked (and what  didn't), and documenting the problems i stumbled into along the way. it's part diary, part technical doc, and maybe a little bit of me talking to myself.

so if you're into seeing how the sausage is made, or just want to laugh at my mistakes, dive into that notebook.

### benchmark
creating a robust benchmark dataset was crucial for evaluating mindex's performance. the process involved generating query-answer pairs using claude (sonnet 3.5), with the full prompt available in `data/claude_prompt.txt`.

key technical aspects of the dataset creation:

1. query generation: aimed for a mix of factual (40%), conceptual (40%), and analytical (20%) questions.
2. answer extraction: verbatim from source documents, typically 1-5 sentences long.
3. diversity constraint: only 30% of queries could directly mention the main contribution of a paper.
4. context simulation: queries framed as if posed to a large knowledge base.
5. coverage: ensured queries spanned different sections of source documents.

the main challenges involved steering claude away from generating queries that referenced specific papers or methods, and balancing query generality with specificity to create realistic scenarios. ensuring extracted answers paired with distinct passages while remaining comprehensive also proved difficult.

the final dataset comprises over 400 samples, split into validation and test sets, covering various ML topics. this benchmark dataset provides a foundation for thorough evaluation and targeted improvements of mindex's performance.

### evaluation
mindex is evaluated using

we employ two complementary metrics:

1. **top-k accuracy**: measures if any of the retrieved chunks contains the answer, using a 0.7 threshold. a score of 1 indicates successful retrieval, 0 otherwise.
2. **mean n-gram overlap score**: calculates the average of the best chunk's n-gram overlap score per retrieval, providing more nuanced results.

a **baseline** implementation was evaluated at 46.24% accuracy with a 0.5093 mean n-gram overlap.

**chunking & cleaning.** we conducted a grid search across chunk sizes 100-800 words, and overlap percentages 20-80%. the search favors larger chunks (not a surprise), which perform better, with optimal overlap ranging from 40% to 60%. the best performer used 800-word chunks with 40% overlap, achieving 63.91% accuracy and a 0.6279 mean overlap score.

800 words is too large a chunk for a search engine result, so we perform a focused grid search in the 500-600 word range with 40-60% overlap. this revealed an optimal configuration of 600-word chunks with 60% overlap (360 words), yielding 62.41% accuracy and a 0.6216 mean n-gram overlap score—a 16-point improvement over the baseline.

attempts at implementing document cleaning within mindex were unsuccessful due to the inherent complexity of parsing PDFs. even seemingly simple tasks, such as identifying paragraphs without visual information, proved to be nearly impossible to achieve reliably. the lack of standardization in PDF structure and the focus on visual layout rather than semantic content made it challenging to develop a universal, simple cleaning strategy that would work across various document sources.

future improvements in mindex will therefore focus on more advanced document parsing techniques. PDF analyzers like surya and custom TeX parsers could significantly enhance our chunking strategy by allowing for more intelligent content extraction based on document structure rather than arbitrary word counts. this is particularly valuable for PDFs, which often lack clear structural indicators, use custom encodings, and may have inconsistent formatting, leading to artifacts in extracted text.

_evaluation is ongoing; this section will be updated with new results._
