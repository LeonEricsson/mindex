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


