# IT Incident Knowledge Assistant

A minimal RAG (Retrieval-Augmented Generation) command-line app. You describe an IT
incident, it retrieves the two most relevant chunks from a small local knowledge base
of `.txt` runbooks, and makes **one** LLM call to produce a structured answer.

If the knowledge base does not cover the incident, it says so instead of guessing.

## Pipeline

```
knowledge_base/*.txt
   -> RecursiveCharacterTextSplitter   (500 chars, 50 overlap)
   -> OpenAIEmbeddings                 (text-embedding-3-small)
   -> FAISS                            (saved to ./faiss_index)
   -> similarity search, top 2
   -> single LLM call                  (gpt-4o-mini)
   -> Problem / Possible Cause / Steps / Resolution / Verification
```

## Project structure

```
simple-rag/
├── app.py
├── requirements.txt
├── .env.example
├── .gitignore
├── knowledge_base/
│   ├── windows_update.txt
│   ├── disk_space.txt
│   ├── services.txt
│   ├── network.txt
│   └── reboot.txt
└── README.md
```

## 1. Install

```bash
git clone https://github.com/<your-username>/simple-rag.git
cd simple-rag

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

Python 3.10 or newer.

## 2. Configure the API key

Copy the template and paste your key from https://platform.openai.com/api-keys:

```bash
cp .env.example .env            # Windows: copy .env.example .env
```

```
OPENAI_API_KEY=sk-...
```

`.env` is in `.gitignore`, so the key never reaches GitHub. Instead of a `.env` file you
can export the variable: `export OPENAI_API_KEY=sk-...`

## 3. Run

Interactive mode:

```bash
python app.py
```

One-shot mode:

```bash
python app.py "Windows server is unable to install a security patch"
```

The first run embeds the knowledge base and saves the FAISS index to `faiss_index/`.
Later runs load that index, so only the chat call costs tokens.
Delete the `faiss_index/` folder whenever you edit the knowledge base.

## 4. Test with three sample incidents

**1. Covered by one document**

```
Windows server is unable to install a security patch
```
Expect: `windows_update.txt` retrieved, with DISM/SFC and SoftwareDistribution steps.

**2. Covered by two documents**

```
Patch installation fails and the C drive has only 500 MB free
```
Expect: `disk_space.txt` and `windows_update.txt` retrieved — this shows retrieval
pulling from more than one source.

**3. Not covered — the guard rail**

```
How do I configure a VPN tunnel on a Cisco ASA firewall?
```
Expect: `Retrieved documents: none` and a message that sufficient information was not
found. No hallucinated answer.

## How hallucination is prevented

1. **Relevance floor** — chunks scoring below `MIN_RELEVANCE` (0.25) are discarded
   before the LLM is called, so off-topic questions never reach it.
2. **Prompt constraint** — the model must answer only from the supplied context and
   return the literal token `NOT_ENOUGH_INFORMATION` when the context is thin.
3. **Temperature 0** — deterministic, non-creative output.

## Tuning

All settings live at the top of `app.py`:

| Setting | Default | Meaning |
| --- | --- | --- |
| `TOP_K` | 2 | chunks retrieved per question |
| `MIN_RELEVANCE` | 0.25 | lower it if valid questions return "not found" |
| `CHUNK_SIZE` | 500 | characters per chunk |
| `CHAT_MODEL` | `gpt-4o-mini` | swap for any chat model |

To add knowledge, drop another `.txt` file into `knowledge_base/` and delete
`faiss_index/` so it is re-indexed.

## Using a different LLM provider

Swap the two LangChain imports. For Anthropic, install `langchain-anthropic`, set
`ANTHROPIC_API_KEY`, and replace `ChatOpenAI` with `ChatAnthropic` — current model
names are listed at https://docs.claude.com/en/docs/about-claude/models. Embeddings can
stay on OpenAI, or move to a local model with `HuggingFaceEmbeddings` if you want the
whole pipeline offline.

## Cost

Roughly 6 KB of text is embedded once (fractions of a cent), and each question sends
about 1,000 tokens to the chat model. Answers are capped at 400 words.
