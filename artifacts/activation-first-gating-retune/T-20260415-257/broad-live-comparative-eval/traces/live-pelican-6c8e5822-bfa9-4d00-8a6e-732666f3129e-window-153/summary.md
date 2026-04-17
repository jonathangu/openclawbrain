# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-153`
- winner mode: `graph_prior_only`
- trace hash: `sha256-ffcf94e58297f053bce53168278403d4ee13aef69fa248575deb3926c6117a0c`
- fixture hash: `sha256-69a203d1bba5e9efdb04c3d2b5eac78a0fd9782e268e61f935bcf93878b096ff`
- score hash: `sha256-933c686ed9ed2f3d938123734371b1910c8297aa92cbf2f98f7cfd1209edcfa7`
- bundle hash: `sha256-d84976f38852335d454b642e511cd21dd34ae22fefc4493412c42141e6fa6cdb`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-d38606df0a6b5cc6fe27f296186c09efc80579f0832811cf6184d8073ca5500a |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-9d51588c0ca22c81eb013000b0451f3ba682381473a68241db867c3a46d596d4 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-10bc1bf0a1d2ba9e56c821c90880f923dbd7116dcda306e725c944d2d6d949fe |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-483d63bd403e665a794aa0420132e22ab9df17ebffe9afc6e72963b21576f8c4 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-04663b33 | sha256-a4a5f6a329c77a7f2cf12ad3012f7a1355161b298a36b47ce433825c3ff5ada5 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-04663b33 | sha256-1dda203b70c604b43dd9421e534d8249c57b5b1070697697303de04db3bb9742 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-0d1d1b94 | sha256-90e1b2b5639cbbfdbadd59030b63dc1f6e6f4617a5a08d88b7788cc51599d260 |
