# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-076`
- winner mode: `graph_prior_only`
- trace hash: `sha256-e90ba91fa2d821b34e7d50d49031d2ca2e725469eba7413ed1eefcf887d0f975`
- fixture hash: `sha256-b66c57ad146f945a1113822081ae1bceec873a0abb858cfb6bafe580d07b22c8`
- score hash: `sha256-b675e1b8f9818ea71e597cab9b3cec29e038b379ccf36743f4909f02e2ae3d77`
- bundle hash: `sha256-412817a7888a3095c52f40b92f5c5b9d0fdf3ff750fa7b98a14ca23c0b6dc93d`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-d7a8f01e83ed8ac33586c073703951c8627b99bf4e9aa0272b865992ce2738f9 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-fd97aced9a23fa7602ddaec1fc5d6682824004952d387a23adace05fa6e776dc |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-9149c346731ddf586682623a38fb22300e4936440194103fe0e2cbd569290a6a |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-b1cf18f6f7214f8c55300c3e9ba02773b1d3bf3c10a6c7bcbf07abb7693c57da |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-3bb7c0c2 | sha256-968c0c2d805ba8591abd57ea7dd38d5949f87c22523f3f0433cf83b644eafbf4 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-3bb7c0c2 | sha256-4d87c668dba8971ac33f1757716a656f0cae8990ecba1df6ab0bf231f8529942 |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-5ff6a877 | sha256-da20ca8d10739ec906cd7ab4a0d259c0f88db72cac196b9e48d779d4f8b344a3 |
