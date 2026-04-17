# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-035`
- winner mode: `graph_prior_only`
- trace hash: `sha256-8ae788de26ca53295ab286d92504e01df10263a019ee2527af469aa665e03d13`
- fixture hash: `sha256-40c450ed66f286026623777e121b2767ec1f98a9a30d5cbc431b359ded23bd1a`
- score hash: `sha256-2d207858aa1dc6a29b12b18654197fe2e5d272197612d68858a7b87dec7b8638`
- bundle hash: `sha256-aaaaa93a21aa427e0c3a434911e2fee30fcebb43a9661792495465035d27e66a`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 100 |
| 2 | vector_only | 100 |
| 3 | learned_route | 40 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 2/4
- phrase hit rate: 0.5

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 1 | 0 | 1 |
| graph_prior_only | 1 | 1 | 1 | 0 | 1 |
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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-1623aa98a961961a182098cbb09dfbf96da5584b9efee0863f57cb38d7ebe41e |
| vector_only | 1 | 1 | 1/1 | 0 | 0 | 1 | 0 | 1 | sha256-92624e82f2ce805ec4b6521fe35ca7a3fd07fd8428ec1f4f076ce927b9531ce7 |
| graph_prior_only | 1 | 1 | 1/1 | 0 | 0 | 1 | 0 | 1 | sha256-d4228c219601ff3585385b0dd6f8d18140d21ca5ab3f9949b6fb1da62ca5a983 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-044a1d60c42fc20ef73bbccde01da1c33110761a6cb430f47a60b85fdb683864 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 100 | yes | 1/1 | no | no | pack-35b361e8 | sha256-254104b2d1c21a99870663aa21d49de0e45ab992fc02187cd2d015058e72fd14 |
| graph_prior_only | turn-1 | 100 | yes | 1/1 | no | no | pack-35b361e8 | sha256-fd41752c28f4b937e84ac563a75e9059419a393ed715622e4f2e4e450573b123 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-5950391d | sha256-4f3996b15ce68836b83a7a20e9c0671999e496aeabf56e6161c80151b584824a |
