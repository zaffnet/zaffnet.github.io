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

Building a chatbot from scratch means handling a lot of plumbing: prompt templates, conversation history, API calls. LangChain does most of this for you, so you can focus on what the bot actually says.

**So what?** Less boilerplate means fewer bugs and faster iteration. You spend time on features, not infrastructure.

![Bar chart showing 29-line vanilla loop vs 12-line LangChain setup](/assets/img/langchain-loc.svg)
<span aria-hidden="true" style="font-size:14px;color:#475569;">A vanilla OpenAI chat loop takes 29 lines; the LangChain version takes 12.</span>

### What LangChain handles for you

LangChain wraps common patterns—prompt templates, memory, retrieval—into reusable components. Instead of building each piece from scratch, you connect them like building blocks. I counted: a basic chat loop with conversation history took 29 lines in plain Python. The LangChain version below does the same thing in 12 lines.

```python
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
chain = ConversationChain(llm=llm, memory=ConversationBufferMemory())

print(chain.predict(input="Walk me through LangChain in two sentences."))
```

Need retrieval? Add a retriever and prompt template without rewriting everything:

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

### Key concepts

* **Chains:** Step-by-step workflows for predictable tasks. Good for FAQs and guided conversations.
* **Agents:** Let the model decide which tool to use next. Useful when the path isn't fixed, but set limits so it doesn't run everything at once.
* **Memory:** Built-in conversation history so the bot remembers context without extra code.

### Tips

* Start simple with a basic chain, then add retrieval or tools as needed.
* Log prompts and steps—LangChain's callback system makes this easy.
* Watch token counts. Abstractions save code, but API costs still add up.

**So what?** LangChain doesn't write better responses—your prompts do that. But it keeps the codebase small enough to understand and maintain. Less plumbing, more building.
