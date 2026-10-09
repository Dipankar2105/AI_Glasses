# Phase 7 — Conversation Orchestration and MCP Foundation

**Status:** Complete  
**Decision:** **ACCEPT**  
**Date:** October 9, 2026  
**Environment:** Python 3.14.6, FastAPI 0.141.1, Pydantic 2.13.4  
**Repository Starting Checkpoint:** `5abf040`  
**Evaluator:** AI Glasses Engineering Team  

---

## 1. Executive Summary & Objective

In **Phase 7**, we designed, implemented, and empirically validated the **Conversation Orchestration Layer** and **Model Context Protocol (MCP)** foundation for NextSight smart glasses.

### Key Objectives Achieved:
1. **Typed Conversation Sessions & History Management:** Thread-safe in-memory session manager with bounded history retention (configurable max window, default 20 messages) and context metadata tracking.
2. **Model Context Protocol (MCP) Server & Tool Registry:** Fully functional, safe MCP tool registry conforming to JSON-RPC 2.0 with strict execution timeouts, argument validation, and error isolation.
3. **Provider-Neutral LLM Interface:** Decoupled `BaseLLMProvider` architecture with an honest `NullLLMProvider` default that explicitly reports `PROVIDER_UNAVAILABLE` rather than hallucinating or simulating AI reasoning.
4. **Non-Negotiable Constraints Maintained:**
   - OCR remains **ON HOLD**; no OCR routing or changes made to frozen Phase 4B vision pipeline (`backend/vision/` has 0 changes).
   - Physical hardware (smart glasses, microphone) is not required; software tests run deterministically.
   - Zero paid or cloud LLM API calls executed.

---

## 2. Architecture & System Flow

```
                                 [Xiaozhi Client / User]
                                            |
                                            v
                      +------------------------------------------+
                      |         FastAPI Conversation API         |
                      |  POST /api/v1/conversation/message       |
                      |  GET  /api/v1/conversation/sessions/{id} |
                      |  POST /api/v1/mcp/jsonrpc                |
                      +---------------------+--------------------+
                                            |
                                            v
+--------------------------------------------------------------------------------+
|                             Conversation Service                               |
|                                                                                |
|   1. Get / Create Session ---------> [SessionManager] (Bounded History Window) |
|   2. Attach Vision Context --------> [Context Metadata]                        |
|   3. Tool Intent / Trigger --------> [MCP Tool Registry]                       |
|                                             |                                  |
|                                             +---> Safe Tools:                  |
|                                             |     - get_system_status          |
|                                             |     - analyze_vision_frame       |
|                                             |     - get_conversation_context   |
|                                             v                                  |
|   4. Generate Response ------------> [BaseLLMProvider]                         |
|                                             |                                  |
|                                             +---> NullLLMProvider (Default)    |
|                                                   "PROVIDER_UNAVAILABLE"       |
+--------------------------------------------------------------------------------+
```

---

## 3. Implemented Endpoints

### A. Conversation Endpoints
- **`POST /api/v1/conversation/message`**:
  - Request: `{"session_id": "...", "message": "...", "tool_to_invoke": "...", "vision_context": {...}}`
  - Response: `{"session_id": "...", "response": "...", "llm_status": "PROVIDER_UNAVAILABLE", "tool_executions": [...]}`
- **`GET /api/v1/conversation/sessions/{session_id}`**:
  - Retrieves full chronological message history and context metadata. Returns `404 Not Found` if unknown.
- **`DELETE /api/v1/conversation/sessions/{session_id}`**:
  - Deletes session from in-memory session manager.

### B. MCP Tool Endpoints
- **`GET /api/v1/mcp/tools`**:
  - Lists registered MCP tools with JSON schemas.
- **`POST /api/v1/mcp/tools/call`**:
  - Directly executes registered MCP tool with argument validation and timeout.
- **`POST /api/v1/mcp/jsonrpc`**:
  - Standard JSON-RPC 2.0 protocol endpoint handling `tools/list`, `tools/call`, and `ping`.

---

## 4. Built-In Safe MCP Tools

| Tool Name | Description | Security / Safety Boundaries |
| :--- | :--- | :--- |
| `get_system_status` | Inspects system health, readiness, and active subsystem statuses. | Read-only. Masks all secrets and filesystem paths. |
| `analyze_vision_frame` | Runs vision pipeline on camera frame / test frame to detect objects and describe the scene. | CPU/thread-pool bounded. Uses frozen Phase 4B pipeline. OCR explicitly marked deferred. |
| `get_conversation_context` | Retrieves message history and context metadata for an active session. | Session-scoped read-only access. |

**Strict Security Constraints:**
- Shell command execution is **PROHIBITED**.
- Unrestricted filesystem and external network requests are **PROHIBITED**.
- Per-tool execution timeout enforced via `asyncio.wait_for` (default 5.0s).

---

## 5. Automated Verification & Test Results

### Focused Test Suite (`backend/tests/test_phase7_conversation_mcp.py`):
19 / 19 passed (100%):
- Session creation, retrieval, bounding (max 20 messages), and deletion.
- MCP tool registration, schema listing, successful execution, unknown tool handling, and timeout enforcement.
- JSON-RPC 2.0 protocol requests (`tools/list`, `tools/call`, `ping`, invalid method handling).
- Honest `NullLLMProvider` reporting (`PROVIDER_UNAVAILABLE`).
- API route integration tests.

### Complete Backend Regression Suite:
```
================== 130 passed, 1 warning in 61.20s ==================
```

---

## 6. How to Run Locally

### Start Server:
```bash
python backend/app.py
```

### Run Phase 7 Tests:
```bash
pytest backend/tests/test_phase7_conversation_mcp.py -v
```

### Run Full Regression Suite:
```bash
pytest backend/tests
```
