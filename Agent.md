# SmartSupply Agentic Replenishment Framework

## 1. Overview & System Architectures (Systems 3 to 6)

The SmartSupply framework provides four LLM-based autonomous replenishment systems operating across a 4-echelon serial supply chain (**Retailer -> Wholesaler -> Distributor -> Manufacturer -> Factory**):

| System | Name | Architecture | Key Characteristics |
| :--- | :--- | :--- | :--- |
| **System 3** | `single_agent` | Single autonomous agent per echelon | Echelon agent decides replenishment orders from local inventory and demand state with prompt v3 guidelines. |
| **System 4** | `multi_agent_info` | Multi-agent with information sharing | Decentralized echelons share upstream/downstream inventory positions, backlogs, and pipeline orders to synchronize supply chain visibility. |
| **System 5** | `multi_agent_tools` | Multi-agent with analytical tools | Specialized CrewAI agents per echelon (**Demand**, **Inventory**, **Procurement**) invoke dedicated analytical tools and aggregate proposals via median or conservative rules. |
| **System 6** | `orchestrated` | Orchestrated negotiation agents | Upstream and downstream agents negotiate orders across contracts, subject to an orchestrator that resolves capacity constraints and deadlocks. |

---

## 2. Tool Modes: Deterministic vs LLM Function Calling

The configuration option `tool_mode` (`deterministic` | `llm_function_calling`, default: `deterministic`) defines how analytical tools interact with LLM agents.

### A. Deterministic Mode (`tool_mode: deterministic`, Default)
- **Documentation Title**: **"Deterministic tools plus LLM decision and negotiation"**.
- **Execution Mechanism**: Analytical tools (`forecast_demand`, `eoq_quantity`, `smoothed_order_up_to`, `validate_order`) are executed directly by deterministic code prior to LLM invocation.
- **LLM Context**: The LLM prompt presents the computed tool outputs:
  - Point demand forecast ($\hat{D}_{t+1}$)
  - Wilson EOQ batch size
  - Smoothed order-up-to target recommendation ($R_t$)
  - Safe order bounds ($[\text{min\_safe}, \text{max\_safe}]$)
- **Agent Role**: The LLM reasons over these deterministic analytical outputs, contextualizes market conditions and operational trade-offs, and decides or negotiates the final order quantity.
- **Reporting & Behavioural Metrics**:
  In deterministic mode, tool-choice accuracy and parameter accuracy are marked `"n/a (tools deterministic)"` in all reports and tables. They are replaced by recommendation adherence metrics:
  1. **Recommendation Adoption Rate**: Share of decisions where the final order is within 5% of the tool recommendation:
     $$|O_t - R_t| \le \max(1.0, 0.05 \times R_t)$$
  2. **Mean Deviation from Recommendation**: Average absolute difference between the agent order and the tool recommendation:
     $$\frac{1}{N} \sum_{t=1}^N |O_t - R_t|$$
  3. **Recommendation Override Share**: Share of decisions where the LLM deliberately overrode the tool recommendation:
     $$\text{Override Share} = 1.0 - \text{Adoption Rate}$$

### B. LLM Function Calling Mode (`tool_mode: llm_function_calling`)
- **Execution Mechanism**: The agent requests tools via structured function calling during prompt execution. The environment executes the requested tool and returns the structured output back to the agent.
- **Reporting & Behavioural Metrics**:
  Tool execution is evaluated against the deterministic **expected-tool rule**:
  - **Demand Agent**: Expected to invoke `forecast_demand`.
  - **Inventory Agent**: Expected to invoke `eoq_quantity` (under normal buffer) or `ss_policy_order`.
  - **Aggregator / Validator**: Expected to invoke `validate_order`.
  - **Tool-Choice Accuracy**: Share of tool calls matching the deterministic expected tool:
    $$\frac{1}{M}\sum_{i=1}^M \mathbb{I}(\text{tool}_i = \text{expected\_tool}_i)$$
  - **Parameter Accuracy**: Share of tool invocations whose arguments match the true underlying state values within 5% tolerance and conform to Pydantic schemas.

---

## 3. Prompt Version 3 (v3) Architecture

All LLM ordering systems (Systems 3 through 6) utilize **Prompt Version 3** (`*_v3.txt`), featuring:

1. **Fixed Instructions First, Variable State Last**:
   - Fixed system directives (echelon role, operational goals, cost structure, stability objective, output schema) appear at the top to optimize prompt caching.
   - Dynamic, step-specific state appears strictly at the bottom.
2. **Order Stability Directive**:
   - Prompts explicitly instruct that **stable orders are part of the objective**:
     > *"Stable orders are part of the objective: Minimizing order variance and avoiding erratic order changes is an explicit operational objective. Maintaining stable, smooth replenishment orders dampens the bullwhip effect and protects upstream echelons from panic oscillations."*
3. **Historical & Tool Signals Included**:
   - Last 5 orders placed: `Last 5 Orders: [o_{t-5}, ..., o_{t-1}]`
   - Forecast demand: `Demand Forecast (t+1): <val>`
   - EOQ batch size: `EOQ Batch Size: <val>`
   - Smoothed target recommendation: `Smoothed Order-Up-To Target (Tool Recommendation): <val>`
   - Feasible bounds: `Safe Order Range: [min, max]`
4. **Output Constraints**:
   - Strict JSON only (no markdown fences, conversational preamble, or extra keys).
   - Reasoning rationale strictly under 40 words.
5. **Prompt Version Tracking**:
   - `prompt_version` (e.g. `single_agent_v3`, `multi_agent_info_v3`, `orchestrated_v3`) is recorded in every row across DuckDB `agent_calls`, `runs_registry`, and `metrics` tables.

---

## 4. Fallback Rule & Resilience

To guarantee non-blocking simulation execution even under LLM failures:
```
[LLM Call Attempt 1]
       │
       ├─► Valid JSON & In-Range Quantity ──► Order Executed (retry_count = 0, fallback = False)
       │
       ▼ (Malformed JSON or Out-of-Range Quantity)
[Retry Attempt 2 with Targeted Correction Prompt]
       │
       ├─► Valid JSON & In-Range Quantity ──► Order Executed (retry_count = 1, fallback = False)
       │
       ▼ (Second Failure / Exception)
[Graceful Fallback to Previous Order (state.last_order)]
       │
       └─► Logged to agent_calls (fallback = True, retry_count = 1)
           Persisted to DuckDB metrics (metric_name = 'fallback_count', metric_value = 1.0)
```

---

## 5. Sample Decision Logs

### Decision Log 1: Deterministic Mode ("deterministic tools plus LLM decision and negotiation")
```
================================================================================
DECISION LOG 1: DETERMINISTIC MODE ('deterministic tools plus LLM decision and negotiation')
================================================================================
System:           single_agent
Echelon:          retailer
Step:             3
Prompt Version:   single_agent_v3
Tool Recommendation: 95.8
Decided Order:    98.0
LLM Reasoning:    "Balanced replenishment aligned with deterministic forecast."
Retries / Fallback: 0 / False
Telemetry Recorded:
  - Recommendation Adopted: True (deviation = 2.2 <= 4.79 tolerance)
  - Tool-Choice Accuracy:   n/a (tools deterministic)
  - Parameter Accuracy:     n/a (tools deterministic)
--------------------------------------------------------------------------------
```

### Decision Log 2: LLM Function Calling Mode ('llm_function_calling')
```
================================================================================
DECISION LOG 2: LLM FUNCTION CALLING MODE ('llm_function_calling')
================================================================================
System:           multi_agent_tools
Echelon:          retailer
Step:             3
Decided Order:    97.5
Tool Execution Trace:
  [1] Invoked Tool: 'forecast_demand ' | Expected: 'forecast_demand ' | Correct: True | Valid Param: True
  [2] Invoked Tool: 'eoq_quantity    ' | Expected: 'eoq_quantity    ' | Correct: True | Valid Param: True
  [3] Invoked Tool: 'ss_policy_order ' | Expected: 'ss_policy_order ' | Correct: True | Valid Param: True
  [4] Invoked Tool: 'validate_order  ' | Expected: 'validate_order  ' | Correct: True | Valid Param: True
Telemetry Recorded:
  - Total Tool Calls:       4
  - Tool-Choice Accuracy:   100.0%
  - Parameter Accuracy:     100.0%
================================================================================
```
