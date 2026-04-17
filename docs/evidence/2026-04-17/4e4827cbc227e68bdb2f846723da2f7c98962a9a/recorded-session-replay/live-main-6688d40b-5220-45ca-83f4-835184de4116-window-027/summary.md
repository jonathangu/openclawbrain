# Recorded Session Replay Proof Bundle

- trace id: `live-main-6688d40b-5220-45ca-83f4-835184de4116-window-027`
- winner mode: `graph_prior_only`
- trace hash: `sha256-3f260af2c7b68b1309e9a87df75f2e99f6d28d47bb3f82fdbd20cd787e51e3c0`
- fixture hash: `sha256-4a50ee1d4a23bf54584481d6c799516fa1f1a51aa4c19299da0f6a6b73848dff`
- score hash: `sha256-3d057a936c6921346e0385bd7a6ebf99f329755b196df357beeae19a07517eea`
- bundle hash: `sha256-13ff7be149d496c2f4bb0f09fec973719842bc2cfc9fdd7c575706400cc800b9`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-9b2a597464226db9617a3470772ef24fd543ab0477b7bbc0a0ad5adf41bc0dc2 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-64cd1a42131d3a4e226d19a902b1da76f945ed2567b75c0cefdbcb05de12e34f |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-dfaae989a832d6f3c21219f730cedb29eff61677df4cc21c9162c442a9f4d106 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-d77ab74c0589c9eeb7890f501f68e415d1a32ba91077ea5dbf17b8c6f1daac0d |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-af169098 | sha256-e5248f5fd40b2304522203f926dc74f5b209892e40a336f6eeaff0e6e7ce26ba |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-af169098 | sha256-fa7857d5e43d40ba6159667fa4ef8433380618889ece31f7c34f10d01a0ae3ee |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-96f3a74f | sha256-b2d4ce0ac3c4e184cffadaff7110fe94de4d6f24aa74233b4c3661bc64517f51 |
