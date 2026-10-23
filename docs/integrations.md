# External integrations requiring credentials, approval or manual setup

No external platform integration has been tested against a live service. Verify each one against the platform's **current** developer terms before enabling it. Until a platform grants access, its adapter reports UNAVAILABLE with a manual search link. It never returns made-up results.

| Integration | Needed for | What is required | Status |
|---|---|---|---|
| **LinkedIn** job search | Discovery | Job search and apply APIs are limited to approved partners (for example Talent Solutions or ATS partners). There is no general public job-search API. Scraping or browser automation of logged-in sessions violates LinkedIn's User Agreement and will not be implemented. | **Built** (Phase 2): reports UNAVAILABLE plus a manual search link; accepts an approved `PartnerFeedClient` |
| **LinkedIn** Easy Apply | Submission | Not available to third parties. Every LinkedIn application is a manual task for the user. | Manual only |
| **Indeed** | Discovery and apply | Indeed's legacy Publisher Job Search API is closed to new publishers. Current programs (partner or ATS integrations) require approval. | **Built** (Phase 2): UNAVAILABLE plus a manual link until an approved client is plugged in |
| **ZipRecruiter** | Discovery | Publisher or partner API by application and approval. | **Built** (Phase 2): UNAVAILABLE plus a manual link until an approved client is plugged in |
| Company ATS: **Greenhouse** Job Board API | Submission | Greenhouse accepts API applications only when authenticated with the **employer's** Job Board API key, so it's usable only for boards whose employer authorizes you (`GREENHOUSE_BOARD_KEYS`). The API has no application lookup, so reconciliation ends in review. | **Built** (`integrations/greenhouse.py`); tested against a simulated server only; **not verified live**; disabled by default |
| Company ATS: Lever, Ashby, others | Submission | Their candidate-submission APIs likewise use employer-issued keys. | Not built; same pattern as Greenhouse if an employer authorizes it |
| OpenAI | LLM | `LLM_API_KEY`, `pip install 'job-agent[openai]'` | Factory built; not exercised live |
| Google Vertex AI (Gemini) | LLM | GCP project, `GOOGLE_CLOUD_PROJECT`, ADC/service-account credentials, `[vertex]` extra | Factory built; not exercised live |
| Anthropic | LLM | `LLM_API_KEY`, `[anthropic]` extra | Factory built; not exercised live |
| PostgreSQL | Persistence and LangGraph checkpoints | `DATABASE_URL` | **Verified** on PostgreSQL 16: migrations, full test suite and the LangGraph checkpointer (`scripts/validate.sh` with `TEST_DATABASE_URL`) |
| Secret manager | Deployed secrets | For example GCP Secret Manager or AWS Secrets Manager, injected as env vars | Not configured |
| Identity provider | API authentication | OIDC/JWT issuer (Phase 7). Dev mode uses static bearer tokens. | Phase 7 |
