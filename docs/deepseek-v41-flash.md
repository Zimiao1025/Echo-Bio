# DeepSeek V4.1 Flash provider profile

Echo-Bio 0.1.0rc1 supports exactly one public live profile:

| Setting | Value |
|---|---|
| Provider | DeepSeek official API |
| Model ID | `deepseek-flash` |
| Base URL | `https://api.deepseek.com` |
| API | Responses API (`POST /responses`) |
| Action transport | provider-native function call |
| Reasoning effort | `none` |
| Automatic retry | disabled |
| Automatic fallback | disabled |
| Credential source | `DEEPSEEK_API_KEY` environment variable |

The default `echo-bio provider-check` is free and sends no network request. `--live` requires the
separate `--confirm-paid` flag and performs at most one request. Echo-Bio never prints the credential.

Check the official DeepSeek pricing and model documentation immediately before a paid run. The
software request limit is not a currency limit because provider prices can change.
