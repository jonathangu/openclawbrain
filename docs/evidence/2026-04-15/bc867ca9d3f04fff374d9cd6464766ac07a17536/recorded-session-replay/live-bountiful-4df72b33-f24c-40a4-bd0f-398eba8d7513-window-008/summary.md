# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-008`
- winner mode: `graph_prior_only`
- trace hash: `sha256-70911ec1e7805ccec970087d6c2246db12da18117e08ef4135c17a78ab963e90`
- fixture hash: `sha256-5f3ce437bc5a34220be72a905054c7058ccdfb9aee9afb407a944b39db8e43dd`
- score hash: `sha256-7c8c314fc553c4717104442f481d39e8b3021725fd08447d9378ff361908b546`
- bundle hash: `sha256-30fb7bc7b65430cfd00b3ec3fa22488117a0b4702671405769d367548209385c`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-c5564d3b460f1097de136a88547e9b2bb9e15503e1a0ceed301551bb8e7b5353 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-3a0dd8be4032ccd6fd1e70420afb18e85e151e4d035027dbe68d13a0456ca426 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-61e248b72e96a12c800144bb7adf5218eba8d7796b69e000a7755fd29fb7dc69 |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-47a1c33d6d8b712e1c9be7a70ec069715a1957c192b9444781cd1a69b33020b1 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-7b4f01e9 | sha256-b0fbc702ea71fe0d01005727c536bfd052b9e13d2f61d0b487dd4f82499c1593 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-7b4f01e9 | sha256-84ce2bfefcba912ccce3ea061481e2c6ef8222de73fba8c9816ed5cdfbca8b1d |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-7b4f01e9 | sha256-b0fbc702ea71fe0d01005727c536bfd052b9e13d2f61d0b487dd4f82499c1593 |
