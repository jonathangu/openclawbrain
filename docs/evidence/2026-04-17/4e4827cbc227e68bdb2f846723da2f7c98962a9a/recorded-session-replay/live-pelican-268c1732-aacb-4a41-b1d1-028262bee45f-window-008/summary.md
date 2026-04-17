# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-008`
- winner mode: `graph_prior_only`
- trace hash: `sha256-821174c9e8199055ec82b211fb2d09a993a3651df37b6c8cbf4707f78e0465ca`
- fixture hash: `sha256-bcd47938b5aaf944e8ae47149b98261af7f5e6b78cefab4ab8c21ef4d0f8288c`
- score hash: `sha256-87e8b95773f1978e7ac9bb83a58b5e5d063dfb4a72ca3973a39fbd4804b1067c`
- bundle hash: `sha256-8c1c6bfa68137189bb0a1787f5c5b3d482ea8b1748e5a4e9e766528b280116b5`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-46bef670c7b3d0d500dbccdbaeb44127bcccbda5425d78ea64b9256410c95a9e |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-2421fbaee4ff38424f77004f48cef79ba6d678307b023727c25aebcfbe13b7da |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-553614911da36d8c0ae5ac8cb0c9bfd9ba3585bb335e336ed0fa9e9620127269 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-e5cab615a39be02ece65dc94833e76cf926ea2ae85ebf91183fa8456a0d598ae |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-e915e8fc | sha256-c6ed49fc20bcd91c5ea15c725339162f63a015cca3ddba6487e4b557ee174f7e |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-e915e8fc | sha256-205b478d064d300643a56c09101c042a488f1f1126d8ef825f679c2801b47fe1 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-00d78169 | sha256-22fe1eab94ad3e780bbe007b75b3670d82c44d50e58befa3c3bcf71241c23105 |
