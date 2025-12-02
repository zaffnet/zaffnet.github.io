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

Last month I rebuilt a chatbot that had grown into a spaghetti monster of API calls, prompt templates, and "temporary" fixes that somehow survived six sprints. The kind of code where you're afraid to change anything because *something* will break, you just don't know what.

Then I rewrote it with LangChain. 29 lines became 12. My weekend suddenly had fewer Slack alerts.

![Hand-drawn sketch comparing lines of code](/assets/img/langchain-loc.svg)

### The "I wrote it myself" trap

When you roll your own chatbot, you end up maintaining everything: conversation memory, prompt formatting, retry logic, streaming... It's like insisting on baking your own bread every morning. Admirable? Sure. Sustainable when you're also trying to ship features? Not so much.

LangChain is basically the bakery. It handles the boring parts—the stuff that's the same for every chatbot—so you can focus on what makes *yours* different.

```python
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
chain = ConversationChain(llm=llm, memory=ConversationBufferMemory())

print(chain.predict(input="Walk me through LangChain in two sentences."))
```

That's a working chatbot with memory. Twelve lines. The vanilla Python version needed 29 just to track conversation history without losing context.

### When you need to get fancier

Most chatbots eventually need retrieval—answering questions from your docs, not just the model's training data. Here's where frameworks really shine. Adding RAG (Retrieval Augmented Generation, for those keeping acronym score at home) is just a few more lines:

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

No need to rewrite your memory handling or prompt logic. You're just snapping a new piece onto the existing chain.

### The mental model

Think of LangChain like LEGO for LLM apps:

* **Chains** = pre-built sequences. Great for predictable workflows like FAQs or onboarding wizards.
* **Agents** = the model picks what to do next. More flexible, but you'll want guardrails unless you enjoy surprise API bills.
* **Memory** = conversation history that just works. No more passing around growing lists of messages.

### A few things I learned the hard way

1. **Start boring.** Begin with a simple chain. Add agents and tools only when the use case genuinely needs them.
2. **Log everything.** LangChain's callback system makes this easy. When (not if) something weird happens, you'll want receipts.
3. **Tokens still cost money.** Abstractions hide complexity, but they don't hide costs. An abstraction that makes five API calls is still five API calls.

The real win isn't the line count—though that's nice. It's that six months from now, when you need to change something, you'll actually understand what's happening. And maybe, just maybe, you'll get to keep your weekend.
