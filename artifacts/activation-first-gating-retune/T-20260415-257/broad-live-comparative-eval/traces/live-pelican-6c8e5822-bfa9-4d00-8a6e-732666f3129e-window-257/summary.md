# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-257`
- winner mode: `graph_prior_only`
- trace hash: `sha256-cc22db3aaa15315761f798aaec1df1acf278bfe86338b981b1d314f80e60f459`
- fixture hash: `sha256-6250124575745297903131838786e09bce6bd0b2285afd782515714f7d74a408`
- score hash: `sha256-1c5ed582bc283c0b054081db4e2984e6852b13b24b531d34b279d0bedefb283c`
- bundle hash: `sha256-2db057f016e76b94513d49c925d2a44f0337213046b96109d5090c70db430814`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 70 |
| 2 | vector_only | 70 |
| 3 | learned_route | 40 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 2/8
- phrase hit rate: 0.25

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.5 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0.5 | 0 | 1 |
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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-363a0ce15ddf219a167f700ba2217552de3446a99d080128478494cb795b929d |
| vector_only | 1 | 1 | 1/2 | 0 | 0 | 1 | 0 | 1 | sha256-93a95966c2ee6ec0c210fc4f0b402875e000fa578b37ae86ed9eb7d196ed1a1b |
| graph_prior_only | 1 | 1 | 1/2 | 0 | 0 | 1 | 0 | 1 | sha256-a5a0bc51d53a43df530af1984b8c8579120a5deaff8b4a4400c4d2d0cc892b72 |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-14e6ce33057c575a03447db65db3b9452aec76be9cd9ab9dc803fba322bdae3f |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 70 | yes | 1/2 | no | no | pack-399c14d9 | sha256-32268c5d0d9bdc500fe4196a597c63bb1d14088fa776111ae348f480e7e19158 |
| graph_prior_only | turn-1 | 70 | yes | 1/2 | no | no | pack-399c14d9 | sha256-0ce20454bc2f7a4f5c20cace48041134479540ad63a29b80463915f46ceab229 |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-b3364998 | sha256-164eee2cff4fea8a67dc1b35828ea6e710196b41ff8c8f683248af59824c244b |
