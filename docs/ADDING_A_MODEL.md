# Adding a New LLM Model

This guide explains how to add, configure, and verify new Large Language Models in the Supply Chain Agents (SCA) framework.

---

## 1. Zero-Code Extensibility

Adding a new model requires **only an entry in [`configs/models.yaml`](file:///c:/Users/saran/smartsupply/configs/models.yaml)**. You do **not** need to modify any Python code. The provider adapters automatically route requests based on the YAML configuration.

Supported adapter types:
- **`openai_compatible`**: Uses the OpenAI Python SDK with a configurable `base_url` and `api_key_env`. Used for:
  - **Isoquant GLM** (e.g., `glm-5.3-flash` at `https://api.isoquant.ai/v1`)
  - **OpenAI** (default base URL `https://api.openai.com/v1`)
  - **xAI / Grok** (`https://api.x.ai/v1`)
  - **Local/Self-hosted endpoints** (vLLM, Ollama, LM Studio)
- **`gemini`**: Uses the official Google GenAI SDK (`google.genai`).
- **`mock`**: Deterministic test double for offline testing.

---

## 2. Model Entry Schema

Each model entry in [`configs/models.yaml`](file:///c:/Users/saran/smartsupply/configs/models.yaml) is validated using Pydantic with strict schema enforcement (`extra="forbid"`):

| Field | Type | Description |
|---|---|---|
| `name` | `string` | Unique identifier for the model in SCA (e.g., `glm-isoquant`, `openai-main`). |
| `provider` | `string` | Adapter family: `openai_compatible`, `gemini`, or `mock`. |
| `model_id` | `string` | Exact model ID expected by provider API (e.g., `glm-5.3-flash`). |
| `base_url` | `string` (optional) | Custom API base URL (required for Isoquant, Grok, local). |
| `api_key_env` | `string` | Name of the environment variable storing the API key. |
| `temperature` | `float` | Default sampling temperature (between `0.0` and `2.0`). |
| `max_tokens` | `integer` | Default maximum tokens per completion. |
| `timeout_s` | `float` | Request timeout in seconds. |
| `price_per_1k_input` | `float` | Price in USD per 1,000 prompt/input tokens. |
| `price_per_1k_output` | `float` | Price in USD per 1,000 completion/output tokens. |
| `supports_json_schema`| `boolean` | `true` for native provider JSON schema; `false` for prompt fallback. |
| `enabled` | `boolean` | `true` to enable for simulations/checks; `false` to disable. |
| `max_concurrency` | `integer` | Per-model bounded concurrency limit (e.g., `10`). |
| `requests_per_minute` | `integer` | Sliding-window RPM rate limit (e.g., `60`). |

---

## 3. The TODO Guardrail

To prevent accidental production errors from placeholder configurations:
- Any model whose `model_id`, `price_per_1k_input`, or `price_per_1k_output` contains `"TODO_FILL_FROM_PROVIDER_DOCS"` or starts with `"TODO"` **must have `enabled: false`**.
- The configuration loader refuses to run any model where `enabled: true` and any TODO field is present.
- Disabled models (`enabled: false`) may safely keep TODO placeholders.

---

## 4. Key Management & Security

1. **Never put API keys in YAML files or code.**
2. Specify only the **variable name** in `api_key_env` (e.g., `ISOQUANT_API_KEY`).
3. Add the actual key to your local `.env` file (copied from [`.env.example`](file:///c:/Users/saran/smartsupply/.env.example)):
   ```bash
   ISOQUANT_API_KEY=iso_your_real_key_here
   ```
4. `.env` is ignored by Git in [`.gitignore`](file:///c:/Users/saran/smartsupply/.gitignore).
5. If a required environment variable is missing, the system raises an error naming the variable (never leaking any key values).

---

## 5. Step-by-Step Examples

### Example A: Enabling `openai-main`
1. Look up the model ID and pricing in OpenAI documentation.
2. In [`.env`](file:///c:/Users/saran/smartsupply/.env), set:
   ```bash
   OPENAI_API_KEY=sk-...
   ```
3. Update the entry in [`configs/models.yaml`](file:///c:/Users/saran/smartsupply/configs/models.yaml):
   ```yaml
   - name: "openai-main"
     provider: "openai_compatible"
     model_id: "gpt-4o"
     base_url: "https://api.openai.com/v1"
     api_key_env: "OPENAI_API_KEY"
     temperature: 0.2
     max_tokens: 2048
     timeout_s: 60.0
     price_per_1k_input: 0.0025
     price_per_1k_output: 0.0100
     supports_json_schema: true
     enabled: true
     max_concurrency: 10
     requests_per_minute: 60
   ```

### Example B: Adding an Ollama / Local Model
Add a new item to `models` in [`configs/models.yaml`](file:///c:/Users/saran/smartsupply/configs/models.yaml):
```yaml
- name: "llama3-local"
  provider: "openai_compatible"
  model_id: "llama3.1:8b"
  base_url: "http://localhost:11434/v1"
  api_key_env: "OLLAMA_API_KEY" # Can be set to 'dummy' in .env
  temperature: 0.1
  max_tokens: 2048
  timeout_s: 120.0
  price_per_1k_input: 0.0
  price_per_1k_output: 0.0
  supports_json_schema: false
  enabled: true
  max_concurrency: 4
  requests_per_minute: 120
```

---

## 6. Verifying Models via CLI

Run the health check command to verify all enabled models:

```bash
python -m sca.llm.check
```

This sends one tiny raw text prompt and one tiny structured-output request to each enabled model, printing a summary table:

```
====================================================================================================================================================
 LLM Model Health Check (configs/models.yaml)
====================================================================================================================================================
model        | status | latency | tokens in/out | cost      | structured output ok | exact error text                                               
-------------+--------+---------+---------------+-----------+----------------------+----------------------------------------------------------------
glm-isoquant | PASS   | 342.1ms | 15/8          | $0.000003 | yes                  | -                                                              
mock         | PASS   | 1.2ms   | 14/5          | $0.000000 | yes                  | -                                                              
====================================================================================================================================================
```

If any enabled model fails, the command prints the exact error text and exits with code `1`.
