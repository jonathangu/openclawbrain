# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-2362908b-54fe-4301-aaaa-003f211ba89c-window-001`
- winner mode: `graph_prior_only`
- trace hash: `sha256-02a1da575e3574b7abbdd906c3a2e763180dabd7aec44faf80aa583f38ef8508`
- fixture hash: `sha256-5054e6a7c0e886819d5cfe411a8be2314f2663654e1f3235054d1d832296503a`
- score hash: `sha256-9e3084b47700311559a30c0e2b071bdd1099c82330dd4b39bf1dd15b3316251b`
- bundle hash: `sha256-0af00d1d1b02d7a1993b3a74adbea6334213d15c55fe5e030ec50b7396c6584b`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 40 |
| 2 | learned_route | 40 |
| 3 | vector_only | 40 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 0/4
- phrase hit rate: 0

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0 | 0 | 1 |
| learned_route | 1 | 1 | 0 | 1 | 1 |

## Hardening Snapshot
- compile failures: 1/4
- compile failure rate: 0.25
- warnings: 5
- promotions: 0

| mode | warnings | compile failures | promotions | export turns | attributed turns |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 1 | 0 | 1 | 1 |
| vector_only | 1 | 0 | 0 | 1 | 1 |
| graph_prior_only | 1 | 0 | 0 | 1 | 1 |
| learned_route | 2 | 0 | 0 | 1 | 1 |

## Mode Table
| mode | turns | compile ok | phrase hits | learned route turns | promotions | export turns | human labels | warnings | score hash |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-f0f3e8434e68958d057893f328bab1455284fa045d15742e91d115a9a3a34202 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-5dddf0e246ef58db49ea255aab36f708b460f19ddce8de9b5d1550aca77182c4 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-bc3ffc6d48ad5c7e541a60c5256ef63cde4e3d44fab3185d531a38db71e7f568 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-849df8528c11872918fb71e7158912a5479ce797536356b194bfe6fc390fa135 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-319e38f5 | sha256-8656a2aaeca6690df153067984a88f9dec3393b30da266a951f8a9b02b63d53e |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-319e38f5 | sha256-8656a2aaeca6690df153067984a88f9dec3393b30da266a951f8a9b02b63d53e |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-94adb1ac | sha256-6f224483d7da5586064d77ecc87ffcf10530a07557bb3e6e1200c8f1ab383d20 |
