# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-153`
- winner mode: `graph_prior_only`
- trace hash: `sha256-ffcf94e58297f053bce53168278403d4ee13aef69fa248575deb3926c6117a0c`
- fixture hash: `sha256-69a203d1bba5e9efdb04c3d2b5eac78a0fd9782e268e61f935bcf93878b096ff`
- score hash: `sha256-b82d398f6caa368b5485dab8e1229cd8bcdf2f9fce1cadd081c949c7f14158c3`
- bundle hash: `sha256-c746884c0a5b2f2cc27fd96f5e8ff9516178c44e668df445de7b2ba872c1c42d`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-d38606df0a6b5cc6fe27f296186c09efc80579f0832811cf6184d8073ca5500a |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-c2bb7bdd7155cd8892defc7af6fabac8e35592affcb984c805137afe0b19d062 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-0002faee5001657f1b161efe4ec7e3caa268072cb1b36c1535257a40ac9484ea |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-e9f1733bc98dfa74c1c748c45b3577616e7b205569f46302594377f3adfc725f |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-164a31a0 | sha256-6b3751daf34f19e53c99b48e4d7454503e89aee215df90495bead818ee0854a1 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-164a31a0 | sha256-087a129323d22ea3a3f6d58501c08578d441c33c23298cbbb967c99f805c6e70 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-164a31a0 | sha256-6b3751daf34f19e53c99b48e4d7454503e89aee215df90495bead818ee0854a1 |
