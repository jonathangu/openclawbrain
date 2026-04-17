# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-ab517e57-6c7d-4bcd-bce3-265ea08c9853-window-004`
- winner mode: `graph_prior_only`
- trace hash: `sha256-e0ad9ec13f7d5b82b36a685375b9d3d24391406d595ab3f8c2b0e0a5247f79c9`
- fixture hash: `sha256-0d1840771e0444519c0d4b5e3c3b57cee2fa58fe3cd78cd2a661af1ba4273a98`
- score hash: `sha256-ef705d13a0abb13893affb7594c08ff3a6a94a8ef972820b29d0121e0b27e4ef`
- bundle hash: `sha256-d71fac483cfc85d2f9a7abca41be0338ff2f9aebdaefe03a0f20f3e36bf8553e`

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
- phrase hits: 0/8
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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-24e5432aabba4b367a9ab9972174d2db006f79b43849cb63eacaea39404c4061 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-684540c2796c129ef8137d4a719bdca4a8969dce0704f789d050d9a385de14ae |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-5a9ce105dc51665064c230eb8a1f12dd5a76b07c370bde3e5fe06074f0381cca |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-4560fa45a876237a7f21323da1b25166cefa81a033f60adcdb8a94f228cc2395 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-d5344667 | sha256-100410d76553d12cce71694ca01218fe2e7540714b58b0f8ab3c0fd66b1c12b3 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-d5344667 | sha256-3cceb0e15fd458144dc1791fecca5938e5fef772809b30f95f1159c229e27370 |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-8ffb6b88 | sha256-1401609a81ccc539d48298feec1f1eec1d02331df9534a398640ddc9fedf9b11 |
