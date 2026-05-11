# SHL Conversational Assessment Agent - Approach Document

## Public API Endpoint
- Base URL: `https://shl-ai-agent-uuks.onrender.com`
- Health route: `GET /health`
- Chat route: `POST /chat`

## 1) Problem Framing and Design Goals
This project implements a stateless conversational recommendation API for SHL assessments. The key goal was to build an agent that is robust under automated replay testing rather than a generic chatbot. The design was guided by hard evaluation constraints:

- schema-compliant JSON on every response
- recommendations strictly grounded in catalog items
- good behavior under multi-turn, correction-heavy conversations
- strong top-k recommendation relevance

The API accepts full message history on each request and returns:
- `reply` (assistant text)
- `recommendations` (0 to 10 catalog items)
- `end_of_conversation` (boolean)

## 2) System Architecture
The architecture is intentionally lightweight and deterministic around critical controls:

1. **FastAPI Routing Layer**
   - `GET /health` for readiness checks.
   - `POST /chat` for conversation flow.
   - Request/response schema enforced by Pydantic models.

2. **Conversation Orchestration Layer**
   - Off-topic and injection filtering.
   - Vague-input clarification path.
   - Domain-specific clarifications for known scenarios.
   - Refinement handling (`add`, `remove`, `drop`) and compare handling.
   - Final confirmation handling (`end_of_conversation`).

3. **Retrieval Layer (TF-IDF)**
   - Catalog documents are indexed from local JSON (`data/shl_catalog.json`).
   - Query built from user history.
   - TF-IDF + cosine similarity retrieves top candidates.
   - Lightweight intent-based score boosts improve ranking in frequent hiring scenarios.

4. **LLM Generation Layer (Groq)**
   - Groq model receives strict system instructions and compact retrieved context.
   - Response requested in JSON object format.
   - Server-side parser validates shape and applies hard safeguards.

5. **Grounding and Safety Guardrails**
   - URL whitelist against catalog.
   - Non-catalog recommendations are stripped.
   - Recommendation list capped at 10.
   - Fallback response for malformed model outputs.

## 3) Why TF-IDF Over Complex Frameworks
This solution uses TF-IDF instead of a heavier RAG stack because the task domain is a fixed, bounded catalog with intent-rich lexical cues (role, skill, seniority, domain terms).

### Why TF-IDF fit this assignment
- Catalog is small and static; no need for heavy ANN indexing.
- Deterministic ranking improves debugability under strict eval.
- Fast runtime and low operational overhead.
- Works well when query text shares direct overlap with catalog metadata.

### Why not LangChain for this implementation
- The flow is straightforward and fully controlled: retrieve -> prompt -> parse -> enforce.
- Adding orchestration frameworks would increase moving parts without clear gain for this catalog size and requirement profile.

### Why no vector database (FAISS/Chroma/pgvector)
- No need for persistent large-scale semantic search infra.
- In-memory TF-IDF retrieval is sufficient and efficient here.
- Avoided extra deployment complexity, ops overhead, and failure points.

## 4) Prompt Design and Policy Strategy
Prompt design was built for evaluation robustness:

- return strict JSON only
- recommend only from supplied catalog context
- ask clarification for vague/underspecified intents
- refuse off-topic and prompt-injection attempts
- handle edits as shortlist refinement, not restart

The agent combines prompt rules with code-level checks, so critical guarantees are enforced beyond prompt compliance.

## 5) Issues Faced and Redesign Decisions
Several issues surfaced during iteration:

1. **Edge case: recommendation too early on underspecified prompts**
   - In some runs, short prompts (role/skill-only) moved to recommendations prematurely.
   - I reviewed provided sample conversations and used Claude support to identify missed behavior patterns.
   - Agent logic was redesigned to improve clarification-first behavior and scenario handling.

2. **Model provider friction**
   - Initial Gemini-based path introduced reliability issues in this setup.
   - Switched to Groq API for more stable structured JSON generation and runtime reliability.

3. **Deployment/runtime dependency mismatches**
   - Resolved through Python/runtime pinning and dependency compatibility checks during Render deployment.

## 6) Evaluation Approach and Improvement Measurement
Testing combined route validation, behavior probes, and scenario replay style checks.

### A) API and schema validation
- Validated `GET /health` and `POST /chat` on deployed endpoint.
- Verified schema fields and types on every `/chat` response.
- Checked invalid-role behavior (HTTP 400).

### B) Behavior probe testing
- Off-topic refusal
- Prompt injection refusal
- Vague first-turn clarification
- Refinement handling (`remove/add`)
- Confirmation-based closure

### C) Groundedness checks
- Recommendation URLs verified against catalog.
- Non-catalog outputs filtered by parser (hallucination control).

### D) Recommendation quality checks
- Multi-turn scenario replay across sample conversation types (public-style traces).
- Improvements measured by reduction in behavior failures and increased recommendation consistency after redesign.

## 7) Hard Eval Alignment
The implemented design targets each required evaluation axis:

- **Schema compliance:** enforced response contract via Pydantic + parser safeguards.
- **Catalog-only recommendations:** URL whitelist and normalization.
- **Turn behavior:** conversation rules encourage convergence and clarification-first flow.
- **Recall@10 readiness:** retrieval + ranking boosts increase relevant candidate presence in top recommendations.
- **Behavior probe pass-rate:** explicit guardrails for refusal, vagueness, refinement, and anti-hallucination.

## 8) Recall@K (Simple Definition)
For a query:

`Recall@K = (relevant assessments appearing in top K) / (total relevant assessments for that query)`

Across N queries:

`Mean Recall@K = (1/N) * sum(Recall@K_i)`

This focuses on whether the final shortlist includes the relevant assessments in top positions.

## 9) Testing Artifacts Used
- Postman collection-based route and behavior testing.
- Deployed endpoint verification on Render.
- Automated end-to-end script: `scripts/e2e_check.py`.

Run command:

`python scripts/e2e_check.py`

This script validates route health, schema integrity, core behavior probes, refinement flow, and invalid input handling in one pass.

## 10) AI Tooling Disclosure
AI coding assistants were used for:
- reviewing sample conversation behavior patterns
- identifying edge-case logic gaps
- accelerating iteration on clarification/refinement behavior
- documentation drafting support

Final safeguards and output constraints are enforced in application logic, not only by prompt text.
