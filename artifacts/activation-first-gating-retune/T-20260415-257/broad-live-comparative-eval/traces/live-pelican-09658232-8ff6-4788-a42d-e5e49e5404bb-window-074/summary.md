# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-074`
- winner mode: `graph_prior_only`
- trace hash: `sha256-145181a4c88521ffe27e5e625ec9bfb16be2cb3d2184f357c331627e92abf897`
- fixture hash: `sha256-bcc383a8935099d2c1130fc4c95751549995a1b863da5b373b03632cf18f4269`
- score hash: `sha256-ec01ee0d6514be24d38a6ed2eb706536ff9ea9cabc1f26d8b55062a15f1d9166`
- bundle hash: `sha256-ba95bd409d00a2e5bad4f6f677102244d2e326397dfc09d39ea97d728d0dd891`

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
- phrase hits: 0/12
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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-4afc95d7380c998fcb03d6870fdcb5302e0b5153c1f87770cfff10ca04ee8cf9 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-58a798bda9df7d4700ba7d95e6b4ac14eec99dcb30da8117515ad4a361ab279c |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-8ef97d9819d7443e0e16daa3b3b5b5dfc8a68cee7118fb744ad7a384f5106d11 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-8522fa9393270ed1ff5af35d6ccd76fd147753ee3fa98cc7b573eace586d6ebf |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-4fe3bbb4 | sha256-8082b8650b1472e0016c91998525e29646dca3a1ffce9d7b26ebc0bb06265a94 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-4fe3bbb4 | sha256-667063b97e2c3fc4de1667f6c9c1044540669658cdccdc78573afe2e92af6a1e |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-3c46eed5 | sha256-382c3a38610ad4a2882a015db4c89156e79612b288924077a6f9466078ccdf7a |
