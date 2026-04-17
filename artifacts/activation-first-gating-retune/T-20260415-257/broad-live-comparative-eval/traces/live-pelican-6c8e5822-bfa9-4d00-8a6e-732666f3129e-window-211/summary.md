# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-211`
- winner mode: `graph_prior_only`
- trace hash: `sha256-c8f3ad7fd7e03d5e6a620d917f9922d423fcf350f433bf42dd24d49c0d04613c`
- fixture hash: `sha256-d090bc75588ff2d651484afffd5d21c674237c8a0eae19ac1a18854f75e95a21`
- score hash: `sha256-55a7cd4c474ea1560e94926a683a67cc065a712e4bd620922b7db7c0de4ad90d`
- bundle hash: `sha256-96f04b6c926876fecb2511a38739a18e9c0a1ef459406cb88b5b8a2cf16aec7a`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 60 |
| 2 | vector_only | 60 |
| 3 | learned_route | 40 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 2/12
- phrase hit rate: 0.166667

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.333333 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0.333333 | 0 | 1 |
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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-a44aad7aa94fbaca9460011cc6ae9061f9cd3a6c6afa137136f8bba1929488be |
| vector_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-05373acfdc83b0f180c4066f5956c6b749bac0bfe2a8d05ea5b178b58743620c |
| graph_prior_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-185ef49e625157efa1b94b0a4151375502cecdf7825f2203e5692120ff6910dd |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-5c721899c6ddb8a8f418fcf5285f8a256491fcc167a40b04eb414b1d62c18509 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | no | no | pack-78e630f2 | sha256-3907eecdd089f9f4404ade51fa761b7f89cc204433e634f658ef107d0f1d2b9e |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | no | no | pack-78e630f2 | sha256-8a4c03cebb3581edbba1f4c704ed9b27de398509f1b10606cf461964a11c5b3a |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-fe966f5b | sha256-e402077b7b3473250e3c1caede279206f4dd887693e437b053a9bcff87b35f98 |
