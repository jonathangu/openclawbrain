# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-035`
- winner mode: `graph_prior_only`
- trace hash: `sha256-8ae788de26ca53295ab286d92504e01df10263a019ee2527af469aa665e03d13`
- fixture hash: `sha256-40c450ed66f286026623777e121b2767ec1f98a9a30d5cbc431b359ded23bd1a`
- score hash: `sha256-6f78693f69714a56e9db71cfc34ed7dffdce97fb2517dbf4a09dcb1d80ce6c65`
- bundle hash: `sha256-ec04b3eb830956311f1d3bcb1f102ff116f35419dab2b8410aceb052535aed36`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 100 |
| 2 | learned_route | 100 |
| 3 | vector_only | 100 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 3/4
- phrase hit rate: 0.75

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 1 | 1 | 1 |
| graph_prior_only | 1 | 1 | 1 | 1 | 1 |
| learned_route | 1 | 1 | 1 | 1 | 1 |

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
| vector_only | 1 | 1 | 1/1 | 1 | 0 | 1 | 0 | 1 | sha256-6a567aa9702c5bc29131cb0097483d7cfd2c245ce4156bf1bf406a9c4a69e853 |
| graph_prior_only | 1 | 1 | 1/1 | 1 | 0 | 1 | 0 | 1 | sha256-03b8f845a61027287cdf0c0834cbc8d0171f024a7e7360c80bd9a4fde1c979d5 |
| learned_route | 1 | 1 | 1/1 | 1 | 0 | 1 | 0 | 2 | sha256-b95cd8024be1b6033db24454702748c4a715101896424c228b4f1c619e413e10 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 100 | yes | 1/1 | yes | no | pack-0dbf8608 | sha256-0f30c64351b3298d679093b901efb159307907feef42d38c1526b8034fc3a1ac |
| graph_prior_only | turn-1 | 100 | yes | 1/1 | yes | no | pack-0dbf8608 | sha256-301550d05bcedc90d96028d3313429d97dfa4ef2fa0d6acd9998d0129d2d22bf |
| learned_route | turn-1 | 100 | yes | 1/1 | yes | no | pack-0dbf8608 | sha256-3bfa933bb3876d9908f2e5c77bdc1dd35267045bc023288d1c8211558619b8c0 |
