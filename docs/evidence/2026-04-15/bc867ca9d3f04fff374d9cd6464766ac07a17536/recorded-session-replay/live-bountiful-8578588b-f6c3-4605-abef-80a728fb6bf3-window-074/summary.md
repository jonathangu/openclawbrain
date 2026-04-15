# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-074`
- winner mode: `graph_prior_only`
- trace hash: `sha256-708994258585f9af49ad6c4184bbdccbe7e42b817caa649c9319156897755b1a`
- fixture hash: `sha256-3753407f1bfe4ff8110a80c454d28c5837a156f1ddd66c296964bb850a56a229`
- score hash: `sha256-2340d2a431a4177d7720934d76dabeef3c09d8f8899dce23ec6fd963fe201f93`
- bundle hash: `sha256-4e9907a04f4abcf2ee176c4c85ea2859ead6992782568b6b04ba937e09228c1d`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-ee2a3369e811726bfb237ce35595dc08f1f6d73159670556f558d53a66965dc0 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-87c5945938669542688365d11668776193ffcdaf1069eb4d4a52e834ef8607f2 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-29266dc60fa4119bd82b4a831b8729f4cba5d8feb5c67c4a1aad879d6ed18dc3 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-77d610fa2c7bcb0d328e934cccdf8bfdf4098766caa097423ea2d75f9955412c |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-797cc78d | sha256-b4ca22b895a667c96c79537569bd37ea4ed3299f5cae05a5e64d334f0fbb6c54 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-797cc78d | sha256-484f1263bf60e9f56490fb3620b75813781c1b59859d779d838db64db512d0a8 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-797cc78d | sha256-b4ca22b895a667c96c79537569bd37ea4ed3299f5cae05a5e64d334f0fbb6c54 |
