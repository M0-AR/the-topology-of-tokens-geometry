# Data card

## Public text corpora

| Name | Upstream | Domain | License/provenance | Split |
|---|---|---|---|---|
| `wikitext_en` | `Salesforce/wikitext`, `wikitext-103-raw-v1` | English Wikipedia prose | Upstream WikiText-103 terms; derived from Wikipedia text | 80/10/10 document hash split |
| `codeparrot_py` | `codeparrot/codeparrot-clean` | Python source files | Upstream CodeParrot terms; GitHub-derived | 80/10/10 document hash split |
| `wikipedia_zh` | `wikimedia/wikipedia`, `20231101.zh` | Chinese Wikipedia prose | Wikimedia terms (CC BY-SA/GFDL family) | 80/10/10 document hash split |
| `wikipedia_hu` | `wikimedia/wikipedia`, `20231101.hu` | Hungarian Wikipedia prose | Wikimedia terms | 80/10/10 document hash split |
| `wikipedia_ar` | `wikimedia/wikipedia`, `20231101.ar` | Arabic Wikipedia prose | Wikimedia terms | 80/10/10 document hash split |
| `wikipedia_sw` | `wikimedia/wikipedia`, `20231101.sw` | Swahili Wikipedia prose | Wikimedia terms | 80/10/10 document hash split |

Selection rule: first 12,000 eligible documents in streaming order, with
documents between 200 and 200,000 characters, exact-text deduplication, and a
deterministic SHA-256 split key. The split is document-level, so no document
appears in more than one split. The upstream dataset revision reported by the
Hugging Face API is recorded in `artifacts/data_manifest.json` when available.

No additional web scraping, annotation, or personal-data enrichment is
performed. Wikipedia and code corpora can contain names, usernames, or
incidental personal data; users should follow the upstream dataset terms and
applicable data-protection rules. The repository redistributes no raw text
beyond cached local files used for reproduction.

## Live market data

- Tickers: AAPL, MSFT, NVDA, JPM, XOM, JNJ, WMT, CAT, KO, DIS, GS, TSLA.
- Windows: 10 years daily, 2 years hourly, 1 month five-minute.
- Primary provider: Yahoo Finance chart endpoint. Daily fallback when Yahoo is
  rate-limited or unavailable: Nasdaq public chart endpoint.
- Raw responses are cached under `data/market/raw/`, hashed in the manifest,
  and excluded from version control. They are not redistributed because the
  providers' terms restrict redistribution.
- Derived records are deterministic text strings containing ticker, timestamp,
  open, high, low, close, volume, and log return. They contain no personal data.
- Splits are chronological. Daily data use train through 2023-12-31, validation
  through 2024-12-31, and test after 2024-12-31. Intraday data use a 60/20/20
  chronological split because their lookback is shorter.
- Survivorship bias, exchange holidays, provider corrections, and
  non-stationarity are acknowledged limitations. Market results are an external
  validity check, not investment advice.

## Derived artifacts

- Tokenizers are fitted only on the training split of each corpus.
- Features are computed only from the training split and the fitted tokenizer.
- Held-out compression and BPB are targets only.
- The market results never enter public-text feature selection or model fitting.

## Execution notes (2026-11-05)

- Yahoo Finance returned HTTP 429 for all requests from the execution host. The
  12 ten-year daily corpora were fetched from the Nasdaq public chart endpoint;
  the 24 intraday windows (2 years hourly and 1 month five-minute) were not
  available from either provider and are recorded as excluded in
  `artifacts/market_manifest.json`. No market values were imputed.
- BPE tokenizers are byte-reversible for all 18 conditions. Unigram tokenizers
  are byte-reversible for 12 of 18 conditions; the six exceptions each contain
  `<unk>` tokens in the held-out test split (`wikitext_en`: one token in one of
  1,218 documents; `wikipedia_sw`: four tokens in one of 1,198 documents).
  `wikitext_en` Unigram 32,768 stops at an actual vocabulary of 29,379.
  `results/tables/tokenizer_byte_audit.csv` records every condition.
