# mindex
a local semantic search engine of your mind index.

[video]

### motivation
i read a lot. papers, blog posts, twitter threads, you name it. a recurring problem i face is recalling information i know i've read somewhere but struggling to find it. to solve this, i built **mindex**. a local semantic search engine for your mind index (where "mind index" refers to things i (you) have read).

this project also gave me a good excuse to gain experience with embedded search. i made the most of this opportunity to learn what works and what doesn't, going to the extent of creating a synthetic benchmark dataset and performing quantitative evaluations of chunking strategies, embedding strategies, and more.

mindex is not a rag system. instead, it provides the exact passage(s) that *best* match your query. while i have little interest in attaching an llm to the end of the pipeline, everything is very extendable if this is something you're interested in!


### usage

**cli**

typical usage would be in form of the cli. to install run

```bash
git clone https://github.com/LeonEricsson/mindex
cd mindex
pip install .
```
now you can initialize a mindex instance 