# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-2362908b-54fe-4301-aaaa-003f211ba89c-window-001`
- winner mode: `graph_prior_only`
- trace hash: `sha256-02a1da575e3574b7abbdd906c3a2e763180dabd7aec44faf80aa583f38ef8508`
- fixture hash: `sha256-5054e6a7c0e886819d5cfe411a8be2314f2663654e1f3235054d1d832296503a`
- score hash: `sha256-7edf22209bf2e78d406e5a9057589d74e1d283e8baa823336c2eef32217958be`
- bundle hash: `sha256-04418b713462323a926f3ff268f4ff6a84103fbdeff0db40bbbd67e87151eb34`

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
| learned_route | 1 | 1 | 0 | 0 | 1 |

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
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-4baf5712caeeb7211f8fffdd854b7dc771c72c81aca9e699ae0925357742a53f |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-8c47b2502c0081ffce15e0a05387f63fddd5eed453c39e60a28f70e056887490 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-268922b593cd6eb0f345e16280f58eef5d0680d5b61080b75f2b589215a78ef7 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-7fe2e016 | sha256-4c888172e0b6a37620d274e62f17bb30e07a10f365d840817f5426733081bab0 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-7fe2e016 | sha256-4c888172e0b6a37620d274e62f17bb30e07a10f365d840817f5426733081bab0 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-7fe2e016 | sha256-4c888172e0b6a37620d274e62f17bb30e07a10f365d840817f5426733081bab0 |
