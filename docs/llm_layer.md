# AI Explainability Layer

`llm_explainer.py` adds a chat-style AI layer on top of CrediX's existing
outputs -- the fraud engine (`fraud_engine.py`), the credit-risk PD model
(`app/model.py`), and the portfolio / risk-lab analytics
(`portfolio_analytics.py`, the dashboard's Risk Lab tab). It does two
things:

1. **Auto-explain** -- turns a result JSON into a short, plain-language,
   professional summary (no ML jargon) for someone who isn't a data
   scientist.
2. **Free-form Q&A** -- answers follow-up questions ("ليه اتعمله رفض؟",
   "why was this flagged?", "what's our current NPL rate?"), grounded
   strictly in the JSON you give it. It never invents numbers that aren't
   in the data.

It replies in whichever language the question was asked in (Arabic or
English), or in a language you force explicitly.

Two front doors use it:

| Front door | Where | What it's for |
|---|---|---|
| **AI Assistant tab** | `dashboard.py`, last tab | Interactive chat for an underwriter looking at one applicant in the Streamlit UI |
| **`POST /explain`** | `api.py` (the unified API on port 8000) | Stateless endpoint any system (core banking, a mobile app, another service) can call with whatever result JSON it already has |

## 1. Get a free API key (2 minutes)

The default provider is **Google Gemini**, because its free tier needs no
credit card and is generous enough for this kind of usage (an
explanation per applicant, plus a handful of follow-up questions).

1. Go to <https://aistudio.google.com/apikey> and sign in with any Google
   account.
2. Click **Create API key** (it can create a project for you
   automatically).
3. Copy the key.

## 2. Configure it

Copy `.env.example` to `.env` if you haven't already, then fill in:

```bash
LLM_PROVIDER=gemini
GEMINI_API_KEY=your-key-here
GEMINI_MODEL=gemini-3.1-flash-lite   # change only if Google renames/retires this model
```

No new Python packages are required -- `llm_explainer.py` reuses
`requests` and `python-dotenv`, both already in `requirements.txt`.

If `GEMINI_API_KEY` is missing, both front doors fail gracefully: the
dashboard tab shows a short setup message instead of crashing, and
`POST /explain` returns HTTP 503 with the same message.

## 3. Using it from the dashboard

Open the **🤖 AI Assistant** tab (last tab) for any applicant. It comes
pre-loaded with:

- that applicant's fraud-engine assessment,
- a live credit-risk score if you've already run `model/train.py` and the
  artifacts exist in `model/artifacts/` (skipped silently otherwise --
  the assistant will say the PD model hasn't been trained yet rather than
  guessing a score),
- a snapshot of the portfolio KPIs computed from the sample core-banking
  workbook.

Click **"اشرحلي النتيجة دي / Explain this result"** for an instant
summary, or just type a question in the chat box in Arabic or English.
The conversation resets automatically when you switch to a different
applicant.

## 4. Using the `/explain` API directly

```bash
curl -X POST http://localhost:8000/explain \
  -H "Content-Type: application/json" \
  -d '{
    "blocks": [
      {"type": "fraud", "data": { "...": "the JSON returned by /api/v1/fraud/evaluate" }}
    ],
    "question": "ليه الطلب اتحول للمراجعة اليدوية؟"
  }'
```

Response:

```json
{
  "answer": "الطلب اتحول للمراجعة اليدوية لأن ...",
  "language": "ar"
}
```

Fields:

- `blocks` (required) -- a list of `{"type": ..., "data": {...}}`. `type`
  is one of `credit_risk`, `fraud`, `portfolio`, or `custom`. Pass
  whichever result(s) you already have; you don't need all three.
- `question` (optional) -- omit it to get an auto-generated summary
  instead of answering something specific.
- `lang` (optional) -- force `"ar"` or `"en"`; auto-detected from
  `question` otherwise (defaults to `"ar"` for auto-summaries).
- `history` (optional) -- prior turns as
  `[{"role": "user"|"assistant", "content": "..."}]`, if you're building
  your own multi-turn chat on top of the endpoint instead of using the
  dashboard's.

## 5. Swapping the LLM provider later

Everything in `llm_explainer.py` except one function is provider-agnostic
(prompt building, context formatting, chat history, language detection).
Only `call_llm()` talks to the network. To switch to OpenAI, Claude, a
self-hosted model, etc.:

1. Rewrite `call_llm()` to hit the new provider's API.
2. Update `LLM_PROVIDER` handling and add the new provider's env vars
   (API key, model name) next to the existing `GEMINI_*` ones.

Nothing in `dashboard.py` or `api.py` needs to change, since they only
call the public `explain_result()` function.

## 6. Notes / limitations

- This is a **decision-support** layer, not a replacement for underwriter
  sign-off -- the dashboard and the system prompt both say so.
- Responses are grounded only in the JSON you pass in. If you ask about
  something not present in that data (e.g. a metric the fraud engine
  doesn't compute), the assistant is instructed to say so rather than
  guess.
- Google's free tier has a daily request quota (see
  <https://ai.google.dev/gemini-api/docs/rate-limits> for current
  numbers). If you outgrow it, the same free-tier key works on the
  pay-as-you-go tier once billing is enabled -- no code changes needed.
