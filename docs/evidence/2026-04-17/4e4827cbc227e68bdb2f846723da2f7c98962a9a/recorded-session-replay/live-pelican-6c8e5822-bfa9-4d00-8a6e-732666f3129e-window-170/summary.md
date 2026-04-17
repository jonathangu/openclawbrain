# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-170`
- winner mode: `graph_prior_only`
- trace hash: `sha256-b236d1de0cb5cdb5c6745d8cf8165eea05f61fe4f12fe91030959e1a1f0a9ef6`
- fixture hash: `sha256-2918c2c3bf776980cb54310652408dcd4b80904c74dc802c02149421011a5050`
- score hash: `sha256-3467d5a2fb7da917f47daebc5e800aaa9ab2e4b0c292a6e4baf9557773006d20`
- bundle hash: `sha256-07bbe8e336a87f8e1cc91a8232dfb06e626521c35976868a663b5b63d6242561`

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
- phrase hits: 0/12
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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-b752caeb990c3acabff60b3183401c5659a9fe06fb13d30bacaaff23a3d4f453 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-332bcce91fd0a1083dfbcce3be1df5c87ab80e46f755cd90094c1ea9606fb381 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-aac798d377bf3e211a9a397e373a8e877f18c15404a3321354b03ec4a47cd679 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-12568498fca02277724816b4deb7671717a1283927a3de700dd54ebe30a49cfe |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-6e464bf4 | sha256-1b0016a726dea5445f05b02d2427041afe86f48bb65dc7101883d1c22474f95b |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-6e464bf4 | sha256-fecd0700c564e2c2b1ac58aa6c98b83d49232ffb8e2108c78ef9f4b26417f709 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-60fc8c79 | sha256-34871dbce1baaf34703ba7599d566d534871a32094c9a5a0b28270e2bbadb828 |
