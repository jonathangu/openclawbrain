# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-211`
- winner mode: `graph_prior_only`
- trace hash: `sha256-c8f3ad7fd7e03d5e6a620d917f9922d423fcf350f433bf42dd24d49c0d04613c`
- fixture hash: `sha256-d090bc75588ff2d651484afffd5d21c674237c8a0eae19ac1a18854f75e95a21`
- score hash: `sha256-a001b5c87002fb959ae61d9de7ef3243ce853f641f6538d0065e2e277347c06b`
- bundle hash: `sha256-c76f5f4d759b8517447bfca20287032c57186f16b38cb9aa80e0f576e2e0c645`

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
| vector_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-9702b4f1c08e5f66db00f5f1ca22276f7984cc09efc17c3e96fe2fc393e4261c |
| graph_prior_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-880c6876675d60442dca10a40f433fd9f199843a44f3895fa14dc490e96505dc |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-e33abd2c9db8927a2b659cf3a8b5d414e156e8b704e3ac9cbab2e7e387df5a23 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | no | no | pack-26d211a9 | sha256-5a802f1a68f32797e56122ef244436d96f11c393fd7ad19b3ba6d31d1427e1af |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | no | no | pack-26d211a9 | sha256-89a95da3ffa5db027cd2d9b3b59049bf8bdd58327f8f8c3cb5b68918a79aa160 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-ac825012 | sha256-54617802967145fccaf8c3497a9ae9ace2d64439fc3dad5d0bde30c94f816796 |
