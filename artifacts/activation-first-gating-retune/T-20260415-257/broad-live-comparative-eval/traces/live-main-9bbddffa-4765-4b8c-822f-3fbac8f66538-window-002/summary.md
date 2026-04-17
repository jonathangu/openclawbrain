# Recorded Session Replay Proof Bundle

- trace id: `live-main-9bbddffa-4765-4b8c-822f-3fbac8f66538-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-d82a353da8dcdcb26be266a0d33a583d117e1f2075d582930b7ead32e3d715fb`
- fixture hash: `sha256-3e1329cac030395635745dbbbafec0f460454aaccbb63b26f155aed0ae65e7c6`
- score hash: `sha256-104877178876a19bac3f98a0baea7ef920d0f3b93dd0df5157a79ec177d99f9a`
- bundle hash: `sha256-85553128e7a7a64e05202b84e187ddec8afeef3ce69e310a7b5332fcf983cf97`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 60 |
| 2 | learned_route | 60 |
| 3 | vector_only | 60 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 3/12
- phrase hit rate: 0.25

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.333333 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0.333333 | 0 | 1 |
| learned_route | 1 | 1 | 0.333333 | 1 | 1 |

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-38cc6030cda1cec211011cfcfcb3fe3c0763917e1cb1cee36ef6155b409ff4f0 |
| vector_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-23456c4ccd0ae705d725c74a88ce18368e750c5cc43b467f60480233ce9f3abd |
| graph_prior_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-47aa2ed773c061a1b06c2b123fc82b4e6b60e0a3ede7e5c9904a78d3d0ec88cb |
| learned_route | 1 | 1 | 1/3 | 1 | 0 | 1 | 0 | 2 | sha256-b33ce6195494529c48d11275b8c14d0de439375824e105c4c7dde56e284c3be8 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | no | no | pack-e17f0814 | sha256-d3cf9c1d7c1f389104b7c14ef8c88b8df85a82bd7b0dd00f72d65c6c8b2ad964 |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | no | no | pack-e17f0814 | sha256-eb44cd06811a71cd2e5ba025baafcdc51a107395562fc61929ef42200d590304 |
| learned_route | turn-1 | 60 | yes | 1/3 | yes | no | pack-8eadbcbf | sha256-84ac400e602e4fc371f202360dced1655e761e3d622aeed13bd03db6869bf2ed |
