# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-2d41cb3b-c723-4429-9992-37a6a6e30bdc-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-4eec8363dbb342027ba768d4c007a5f6bb26136392615c736348968a0e88b605`
- fixture hash: `sha256-dd05986903c2d3b37c7fdf438fab0c6737ffb4bb4a24103981b21ff15a79f4c7`
- score hash: `sha256-a745b1a77d932272943e49b7b2cd91b98f83d7bbecef5b5dc78feb04cd432363`
- bundle hash: `sha256-a1ab4c941721f12297c6a9d9aa412b70896b4a997a89cd270a8f0a54b03af2b3`

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
| vector_only | 1 | 1 | 0 | 1 | 1 |
| graph_prior_only | 1 | 1 | 0 | 1 | 1 |
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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-1952d2df118914f878f9222de732246d88461cd2b6a05bddbf4ef8392b473715 |
| vector_only | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 1 | sha256-04916a440a25542f085f5c3ba3dceb35f20c5e151f9ba60273e3adad710d06ec |
| graph_prior_only | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 1 | sha256-2f56b3e54bbee4aff9bfaf820f34e8e662c9ea6061492b838f78546fc3e76106 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-76fe6503e5db67c57a2442c63dba43e1581874bb4a831dd3f4bd07ec77a8d7ca |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | yes | no | pack-bde11863 | sha256-82929c1e645b698654bacc42cbc25583a794ba600e553bd9275caf8713b006dc |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | yes | no | pack-bde11863 | sha256-b65244fb957b18bddbd96c24caa8c3597bcdbf0bc86398314da5fe665366d0c5 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-bde11863 | sha256-efc5cdd98b6d37ecb5c9dcbb17eec41a15b2cf52e4c8852f8508e5ba011f2f93 |
