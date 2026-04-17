# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-081`
- winner mode: `graph_prior_only`
- trace hash: `sha256-404edb8e8148990f7dd9ba6ee7f25c05f0fb22e6cb89bbcabd63ecc3b578e01f`
- fixture hash: `sha256-1b74e2bda5a418c592958218e18909375ff9b60c95cbb866c13aa0c8c5768f8f`
- score hash: `sha256-e732c17d1d88efc6ef2eec514e77cdf29c05bbeddb8043da4b73a80bab0e1d7d`
- bundle hash: `sha256-6d1d6a23fd2c0ab66a438ff3d44c85d4bc79fb370ac7c4ebcfb7fa2da3868bf3`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-d8d61d87f3931e7b37ce8220e425e901452b419dbbce6c76196690cf92892dce |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-4457c47e033b341f197759bd7eda78ba1d205409eed1822429f5892be8489536 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-60db7c7675226de23c39769349879c6aeb8fb6b1e9724df255da34bc85aa6acd |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-60c0a22318175a26cc81f7ca4acab566b681f08c6ce1c9e5b87c7046ae70bdd4 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-f9fb71c6 | sha256-a176ff897b584dbab70b80b42ec2a70fd47c853882c4c90decc2b4af8e9ecbc4 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-f9fb71c6 | sha256-847c38a4a533d417a25c34e2fd1a628eace17bfcf50bee03b357960505972fd2 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-ed08939d | sha256-cf4f12ce6a94cbab087e74d812e5882f46c2c6cb7fbe9c931304eed238e38b9f |
