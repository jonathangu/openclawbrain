# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-063`
- winner mode: `graph_prior_only`
- trace hash: `sha256-de0740d5fcaea5454096d093094f34c04b3f0e1916a63c31d443d2fb518007d1`
- fixture hash: `sha256-fe0c7d701266203a3973486da92812ab1d527b01c29d86dc3ebbae41ae89dfcf`
- score hash: `sha256-2c4f4ddd0b0999b4f533fbf6b4813493f192a016c527b73fd6fcbbca74e8217e`
- bundle hash: `sha256-3ec17c779d9e8e476bb3ecff26ace56d9825d15d05539c8cc53d7b3a93303949`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-aec7ec8f6799a0ee01f6b4130aafafed76ac2a835827ba3dac19bfdac983b407 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-6f4115507057d519a3a56af6bfdaa288f961cfd46d8c1d7475acb953e58775a9 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-1ed8a343217aa98f13ad06bfadb409be1593ecd0fab40a110696b4a467883205 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-5c238b5c25a01b1432a7ec2e295e3da5cdfaaf93910e008924749b0950c51a48 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-1838a174 | sha256-34d891e2ceeaa402fb5a9d75f0af0497f492e9a4d8a99c001d2b8756199c859b |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-1838a174 | sha256-a6ba700042d83e2bd44fba94188fff12c2c451386e6433a04abf26714032859e |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-7fdb36ad | sha256-625eedada62c070c44931cd63d5a82a8e0932c22fe47f126439de70f1841077c |
