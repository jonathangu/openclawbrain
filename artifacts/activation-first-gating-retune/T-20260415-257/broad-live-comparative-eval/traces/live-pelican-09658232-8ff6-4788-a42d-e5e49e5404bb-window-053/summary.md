# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-053`
- winner mode: `graph_prior_only`
- trace hash: `sha256-ebe19ca9bc459ccc52f05cb3ef8e24277b70f060cd7939f534aee99c63488ae5`
- fixture hash: `sha256-83e33e2dc3d5736fb8b475959b3f799a1522431a9ceb8bc4c7fc74edb18967c0`
- score hash: `sha256-988c30734b270d85a0b06a661819ea97bb0962fb3f9a7345fe42a3f1693e5f51`
- bundle hash: `sha256-6d667767ed40410294f99ecd38c0a28381e281237231f0d7d616e1c1f5727fe5`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-38dbfef9dbd8d3a664a1f95db7f92b5a52579781d0eb6bee8aa47758b54b5ce0 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-473a4774828583a82f4b3d002b497dc803d7129b49626ffdac00fe8bdb6d9558 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-30e563a63a50960c20eb0d2a7f5d8fb965a0f058df30d69cfaba5ec78c4c6899 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-07d8077848896260b4db8a9922b38967ca373a39331360f70b67b1de86e8c10a |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-02819013 | sha256-83dfd0dc7654b799c0ea461c507850d535740e930e554301398deabdac52e9e9 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-02819013 | sha256-917462f3d53571034405aaa3fafcf9e0e6a2073369897b75eff6c4d3e4e336a6 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-2cbd34d6 | sha256-304b34061a87a07a40375634bfca68cca46d927867055aed9f1d926bf0ceb157 |
