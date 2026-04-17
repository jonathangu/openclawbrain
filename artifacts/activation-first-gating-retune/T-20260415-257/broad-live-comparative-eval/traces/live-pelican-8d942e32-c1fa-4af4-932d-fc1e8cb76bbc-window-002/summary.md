# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-8d942e32-c1fa-4af4-932d-fc1e8cb76bbc-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-64a26e8f5e6980d9246d302acab2d34c4e18ddd7be07096a6ca889aa90e2228a`
- fixture hash: `sha256-833c77206f16af416cd188d9e8ee18c5e59708b98a4500bfd6d7d22e62fa078a`
- score hash: `sha256-0bf4452e35b055dfbfc2a06c9055a250d5de6e31036d33e90d83cb6c0a90ccc0`
- bundle hash: `sha256-98bb0015714e6b1a61f4778e42cea43db21b090651c32883eb044be60647d675`

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
- phrase hits: 0/8
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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-947e512b0a82ec6d517ce602229a8e508d29ae58b836a4631a42b14c828dead3 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-661a8965747baf31b2500a09ef0fb6f5c60e91f898821b2f9bd7da409851a6d7 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-13440d8db919dcf31811c1897e5b0f98df98bec0fb515b1fa91658ffcb587f26 |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-40e9c7db179e60af8cab4836389f8b6bd7e2e9d3cd096f0b3c178c956863fb45 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-901d70bd | sha256-ab9574bf9c31dc55e5232182b20e32f9918347d1fa1d348d7c24934a79b2c293 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-901d70bd | sha256-156a2c89a1bdc8bb878404de21cda0fa98b45320a7f19efc05715c4819e86466 |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-8b19bc74 | sha256-b8b8b646debdc0acf83b61b04081d3b8e3550138a4202b0ae0dbf74f2847c3ff |
