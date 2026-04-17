# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-081`
- winner mode: `graph_prior_only`
- trace hash: `sha256-bed00a4e07440963fcb335df85735aa4cdf299e8eeeee26d6072510f9d967592`
- fixture hash: `sha256-7b9b3de84eea4b8f6489862313ae3b9c5b0de1ba49de793c86ffbf0e24eac4d6`
- score hash: `sha256-95a0a2d1cdc2bb8815a8eeb9189d1f3dfd3a595e59165960f89a513c5f645967`
- bundle hash: `sha256-881472d3be518c60d8d26bd385e2af5167c5cb96cda94e43eee1ecbdcf913c27`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-7244838abdb700b7974c4fb03ae1270d3910510dcc5c175db289b9a82a5df872 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-842d64a1a405842e6bff82f312beb42965f39467bc8fbd1b46862ce88510362a |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-fd5bec8d7717bec4cb8c182a61294be97ab24f27cd347f0eb919f61a70d79a50 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-2ff8f48ff6374c8d1672d0a9a6fa458c5e9e0ef3787b6a32ee6d4f820005ca7a |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-488b16b9 | sha256-72b23ca35252cedca3fc53c38f4e34c022b4501cf53a9e49c3bfec960e722ff0 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-488b16b9 | sha256-c8ec60db189917e794c069a218775061f070a1fe47e3091aef801f1f71005a75 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-0721f656 | sha256-5832ae5a0ff4b226e741bacfc43bdb29bf859fbc3385a58996893b8f6c927e7d |
