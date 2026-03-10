# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

#### Weighted-Router Alias Model Resolution
- **Stable model aliases**: Use simple names like `glm-5`, `kimi-k2.5` instead of provider-specific IDs
- **Automatic provider selection**: Weighted random selection distributes load across available providers
- **Provider weights**: ollama_cloud (70%), opencode_go (15%), chutes (15%)
- **Supported aliases**:
  - `glm-5` → ollama_cloud/glm-5, opencode_go/glm-5, chutes/zai-org/GLM-5-TEE
  - `kimi-k2.5` → ollama_cloud/kimi-k2.5, opencode_go/kimi-k2.5, chutes/moonshotai/Kimi-K2.5-TEE
  - `qwen3-coder-next` → ollama_cloud/qwen3-coder-next, chutes/Qwen/Qwen3-Coder-Next-TEE
  - `minimax-m2.5` → ollama_cloud/minimax-m2.5, opencode_go/minimax-m2.5, chutes/MiniMaxAI/MiniMax-M2.5-TEE
  - `qwen3.5` → ollama_cloud/qwen3.5, chutes/Qwen/Qwen3.5-397B-A17B-TEE
  - `deepseek` → ollama_cloud/deepseek-v3.2:cloud, chutes/deepseek-ai/DeepSeek-V3.2-TEE

#### Enhanced Credential Exhaustion Handling
- **Mid-stream exhaustion detection**: Real-time monitoring of streaming chunks for quota exhaustion signals
- **HTTP 503 Service Unavailable**: Proper status code when all credentials are exhausted
- **Structured error responses**: Clients receive detailed exhaustion information for better retry strategies
- **New exception**: `AllCredentialsExhaustedError` for definitive exhaustion scenarios

#### Improved Cooldown Management
- **Exponential backoff**: 1s base delay, 300s max delay (prevents thundering herd)
- **Jitter support**: ±50% randomization to stagger provider recovery attempts
- **Better rate limit handling**: More graceful recovery under load

### Technical Details

#### Commit History
- `0b8903b` feat(cooldown): add exponential backoff with jitter for provider re-enable
- `b50de84` feat(exhaustion): improve credential exhaustion detection and HTTP response
- `f140200` feat(weighted-router): add alias model resolution with weighted provider selection
- `a3389a0` fix(weighted-router): remove unsupported models from OpenCode GO alias map

## Usage Examples

### Using Weighted-Router Aliases

**Before (provider-specific):**
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer $PROXY_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ollama_cloud/glm-5",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

**After (stable alias):**
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer $PROXY_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "glm-5",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

The router automatically selects an available provider based on weighted random selection.

### Handling Exhaustion Errors

When all credentials are exhausted, clients now receive HTTP 503:

```json
{
  "error": {
    "type": "service_unavailable",
    "message": "All credentials exhausted for provider",
    "details": {
      "provider": "ollama_cloud",
      "reset_time": "2026-03-10T04:00:00Z"
    }
  }
}
```

This enables clients to implement intelligent retry strategies with proper backoff.

---

## Previous Releases

See [GitHub Releases](https://github.com/Mirrowel/LLM-API-Key-Proxy/releases) for historical changelog entries.
