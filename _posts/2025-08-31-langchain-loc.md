---
layout: post
title: "Using LangChain to Reduce LOC of a Chatbot"
desc: "A look at how the LangChain framework can simplify chatbot development and reduce lines of code."
keywords: "LangChain, Chatbot, Python, LLM"
date:   2025-08-31 18:53:04.426237
lastmod: 2025-08-31 18:53:04.426237
comments: true
permalink: langchain-loc
---

Building a chatbot from scratch can feel like assembling IKEA furniture without the little hex key: doable, but you'll invent new words along the way. LangChain trims the drama by handling the plumbing (prompt templates, vector stores, conversation memory) so you can focus on the parts that actually delight users.

![Toy bar chart comparing LOC](/assets/img/langchain-loc.svg)

![What the 12-line stack wires together](/assets/img/langchain-stack.svg)

### The power of abstraction

LangChain wraps common LLM patterns so you don't have to rebuild them. Instead of wiring up every API call by hand, you stitch together chains and agents like Lego bricks. The result: fewer lines, less boilerplate, and more time to argue about your bot's personality. I counted: a minimal OpenAI-only chat loop for turn-based history took 29 lines, while the LangChain version below needed 12 lines to keep the same state.

```python
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
chain = ConversationChain(llm=llm, memory=ConversationBufferMemory())

print(chain.predict(input="Walk me through LangChain in two sentences."))
```

The new diagram captures what those 12 lines wire together: a prompt template, a ChatOpenAI client set to gpt-4o-mini, and a ConversationBufferMemory block that keeps the transcript tidy without extra code.

Need retrieval? Swap in a retriever and a prompt template without rewriting half your app:

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA

prompt = ChatPromptTemplate.from_template(
    """Use the snippets below to answer:
    {context}
    Question: {question}
    """
)
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=my_vectorstore.as_retriever(),
    chain_type_kwargs={"prompt": prompt},
)
response = qa.invoke({"question": "How do we tune alpha?"})
```

### Chains, agents, and fewer footguns

* **Chains:** Straight-line workflows for predictable tasks. Great for FAQs and onboarding flows.
* **Agents:** Let the LLM decide which tool to call next, like a choose-your-own-adventure but for API calls. Just give it guardrails so it doesn’t binge every tool at once like it's buffering a full season on Netflix.
* **Memory:** ConversationBufferMemory or summary memory keeps context tight so you don't repeat yourself. Your future self will thank you when debugging logs.

### Practical tips

* Start with a basic chain, then sprinkle in retrieval or tools as you validate user demand.
* Log prompts and intermediate steps; LangChain's callback system makes this straightforward.
* Keep an eye on token counts, because abstractions save LOC but the meter still runs.

LangChain won't make your bot write Shakespeare (unless you ask nicely), but it does keep the codebase lean enough to fit in your mental cache. Less boilerplate, more shipping.
