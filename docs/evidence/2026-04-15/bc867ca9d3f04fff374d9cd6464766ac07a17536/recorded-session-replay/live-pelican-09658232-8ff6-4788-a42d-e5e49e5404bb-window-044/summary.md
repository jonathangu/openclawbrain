# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-044`
- winner mode: `graph_prior_only`
- trace hash: `sha256-0ad01cba10197b8b05f27f973f11a9d5ae36ed08ae5a28bf29003ec749435fdb`
- fixture hash: `sha256-6830cd222cfba386b49de3a4d46620d84d8c333ee746eafd9c6c6f8ee2dfa95c`
- score hash: `sha256-1e6e1609a765f8e2c1674041467debd446c3c49cf9ef0425d00ecb2633bbd316`
- bundle hash: `sha256-db960ffee6dbcb848ea08d181d36ba175bb676433af3dee26c608b028e3f2d82`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-490c3e2fb439b8787ca9fe3cb573a9c587cc3abe414e2faf637ea2b84d91d268 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-9af446711a6145a543c44fa378ec20a6f0d151e3168276195a96673b036ca5ee |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-ad056128009f0bf05a5469d06f7e152f718f55c13d3d677b074b22aa05d3ca62 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-5a1eba333fb3455265626c85c263f6d7c232c6c92edcda8ca925ab9557d7f1c0 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-1eee51bf | sha256-3dcb9c18e8fe61e49d6a1f735089e5b2667c5456c16eb20eb8971f07563f1e08 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-1eee51bf | sha256-23f0a71c2f9f72abfa4c0acdca3a995558367d87f389170063b3e3256af94a66 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-1eee51bf | sha256-3dcb9c18e8fe61e49d6a1f735089e5b2667c5456c16eb20eb8971f07563f1e08 |
