# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-062`
- winner mode: `graph_prior_only`
- trace hash: `sha256-68dd4afa8cf6968b41418bd460fed5641fe37e0c30a004be1adb6fd97d678410`
- fixture hash: `sha256-0984347c035679f491e5e5ce92160de0970752142af6bd7d0f80779707ccfa84`
- score hash: `sha256-08f6fa2b433d7f00457a5719aa6395f04b026894a3289ea7f57fc8f7af58909f`
- bundle hash: `sha256-5d861d6527ac820fa68dadabc514f02dcf520e5ab106260726fa5ef8a625e16b`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-4e7043d44034a818b042ec107f761c7c9e4d805591027e32242d8b764dc9d866 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-bd71a8955ff2a0cfaf89937ab3085beecf2266f57a946492804d8ad207528f57 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-feee2dce4f81c60e0aee88fbdd9a9c312101577b20f92e3f1f893c5b2e7b96c0 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-e46afbf2e681c984890d64d4073fb0b08884aad8af45d799be7b2d83d29f8088 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-eaad435a | sha256-e9a6c74a99114d348dcbdadab4f0bd3be4633017c947bdbb9eed3a04a1442114 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-eaad435a | sha256-751593de71231b1f09a05fbae956bc6f6a53bd80b1a247c3448723da37dd02d8 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-f8c6dab7 | sha256-d36fbebebaad6ee14381b983b8e733da3c282bbc4c009b79d85de33446286834 |
